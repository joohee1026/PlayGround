import time
from pathlib import Path

import wandb
import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_

from model import get_model
from utils import SetOptimizer, SetScheduler


class Solver:
    def __init__(self, cfg, loader, fold=None):
        self.cfg = cfg
        self.train_loader, self.valid_loader = loader
        
        self.device = cfg.base.device
        self.model = get_model(
            cfg.model.name,
            cfg.model.pretrained
        )
        self.model.to(self.device)
        self.wce_weights = self.get_wce_info()

        self.prep_training()
        if fold is None:
            self.prefix = "HOLDOUT"

    def train(self):
        start = time.time()
        best_acc, best_epoch = 0., 1
        for epoch in range(1, self.n_epoch+1):
            epoch_start = time.time()
            trn_loss, trn_acc, trn_loss_t, trn_acc_t  = self.update(epoch)
            val_loss, val_acc, val_loss_t, val_acc_t = self.validation(epoch)

            trn_loss_d, trn_loss_g, trn_loss_e = trn_loss_t
            trn_acc_d, trn_acc_g, trn_acc_e = trn_acc_t
            val_loss_d, val_loss_g, val_loss_e = val_loss_t
            val_acc_d, val_acc_g, val_acc_e = val_acc_t

            if self.cfg.scheduler.name in ("ReduceLROnPlateau", "LinearLR"):
                self.scheduler.step(val_loss)

            if (val_acc > best_acc) and self.cfg.base.save_dir:
                logger.info(
                    f"Improved from {best_acc:.3f}({best_epoch} epoch) to {val_acc:.3f}({epoch} epoch)"
                )                
                best_acc, best_epoch = val_acc, epoch
                model_path = Path(self.cfg.base.save_dir) / "model" / "model.pt"
                torch.save(self.model.state_dict(), model_path.as_posix())
                logger.info(f"{epoch} Epoch model saved")
        
            elapsed = time.time() - epoch_start
            logger.info(f"EPOCH {epoch} complete in {elapsed:.0f}s")
            logger.info(f"TRAIN LOSS (TOTAL/D/G/E): {trn_loss:.3f}/{trn_loss_d:.3f}/{trn_loss_g:.3f}/{trn_loss_e:.3f}")
            logger.info(f"TRAIN ACC  (TOTAL/D/G/E): {trn_acc:.3f}/{trn_acc_d:.3f}/{trn_acc_g:.3f}/{trn_acc_e:.3f}")
            logger.info(f"VALID LOSS (TOTAL/D/G/E): {val_loss:.3f}/{val_loss_d:.3f}/{val_loss_g:.3f}/{val_loss_e:.3f}")
            logger.info(f"VALID ACC  (TOTAL/D/G/E): {val_acc:.3f}/{val_acc_d:.3f}/{val_acc_g:.3f}/{val_acc_e:.3f}")

            if self.cfg.log.wandb:
                lr = self.optimizer.param_groups[0]["lr"]
                wandb.log({
                    "Train Loss": trn_loss,
                    "Train Loss D": trn_loss_d, 
                    "Train Loss G": trn_loss_g, 
                    "Train Loss E": trn_loss_e,
                })
                wandb.log({
                    "Train Acc": trn_acc,
                    "Train Acc D": trn_acc_d, 
                    "Train Acc G": trn_acc_g, 
                    "Train Acc E": trn_acc_e,
                })
                wandb.log({
                    "Valid Loss": val_loss,
                    "Valid Loss D": val_loss_d,
                    "Valid Loss G": val_loss_g,
                    "Valid Loss E": val_loss_e,
                })
                wandb.log({
                    "Valid Acc": val_acc,
                    "Valid Acc D": val_acc_d,
                    "Valid Acc G": val_acc_g,
                    "Valid Acc E": val_acc_e,
                })
                wandb.log({
                    "Learning Rate": lr
                })

        elapsed = time.time() - start
        logger.info(f"Complete in {elapsed:.0f}s")
        logger.info(f"Best Acc: {best_acc:.3f} ({best_epoch}/{self.n_epoch})") 

    def prep_training(self):
        self.n_epoch = self.cfg.train.n_epoch
        self.amp = self.cfg.train.amp
        self.clip = self.cfg.train.max_grad_norm

        if self.cfg.loss.wce:
            self.criterions = [
                nn.CrossEntropyLoss(weight=w) for w in self.wce_weights
            ]
        else:
            self.criterions = [nn.CrossEntropyLoss() for _ in range(3)]
        self.wd, self.wg, self.we = self.cfg.loss.task_weight
            
        self.optimizer = SetOptimizer(self.cfg.optimizer, self.model).load
        self.scheduler = SetScheduler(self.cfg.scheduler, self.optimizer).load

    def get_loss(self, logits, labels):
        losses = [
            self.criterions[i](logits[i], labels[i]) for i in range(3)
        ]        
        return self.wd*losses[0] + self.wg*losses[1] + self.we*losses[2], losses

    def get_n_correct(self, logits, labels):
        predictions = [torch.max(logit, axis=1)[1] for logit in logits]
        labels = [torch.max(label, axis=1)[1] for label in labels]
        crts = [(pred == label).sum().item() for pred, label in zip(predictions, labels)]
        return crts

    def update(self, epoch):
        self.model.train()

        losses, losses_t = 0., [0. for _ in range(3)]
        total, crt_list = 0, [0 for _ in range(3)]
        progress = tqdm(
            enumerate(self.train_loader), ncols=200,
            desc=f"[{self.prefix}]-[EPOCH {epoch}/{self.n_epoch}]",
            total=len(self.train_loader)
        )
        for i, (image, label_d, label_g, label_e) in progress:
            image = image.to(self.device, dtype=torch.float)
            label_d = label_d.to(self.device, dtype=torch.float)
            label_g = label_g.to(self.device, dtype=torch.float)
            label_e = label_e.to(self.device, dtype=torch.float)
            
            self.optimizer.zero_grad()
            scaler = amp.GradScaler() if self.amp else None
            if self.amp:
                with amp.autocast(enabled=True):
                    logits = self.model(image)
                    loss, (loss_d, loss_g, loss_e) = self.get_loss(
                        logits, (label_d, label_g, label_e)
                   )   
                scaler.scale(loss).backward()
                if self.clip:
                    scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.clip)
                scaler.step(self.optimizer)
                scaler.update()

            else:
                logits = self.model(image)
                loss, (loss_d, loss_g, loss_e) = self.get_loss(
                    logits, (label_d, label_g, label_e)
                )
                loss.backward()
                if self.clip:
                    clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                
            if (self.scheduler is not None) and \
                self.cfg.scheduler.name not in ("ReduceLROnPlateau", "LinearLR"):
                self.scheduler.step()
            
            losses += loss.item()
            losses_t[0] += loss_d.item()
            losses_t[1] += loss_g.item()
            losses_t[2] += loss_e.item()            
            train_loss = losses / (i+1)
            train_loss_t = [lst/(i+1) for lst in losses_t]
            
            total += image.shape[0]
            crts = self.get_n_correct(logits, (label_d, label_g, label_e))
            crt_list[0] += crts[0]
            crt_list[1] += crts[1]
            crt_list[2] += crts[2]
            train_acc_t = [c/total for c in crt_list]
            train_acc = np.mean(train_acc_t)

            lr = self.optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                LR=lr, LOSS=train_loss, ACC=train_acc,
                LOSS_D=train_loss_t[0], LOSS_G=train_loss_t[1], LOSS_E=train_loss_t[2],
                ACC_D=train_acc_t[0], ACC_G=train_acc_t[1], ACC_E=train_acc_t[2]
            )
        return train_loss, train_acc, train_loss_t, train_acc_t

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        
        losses, losses_t = 0., [0. for _ in range(3)]
        total, crt_list = 0, [0 for _ in range(3)]
        progress = tqdm(
            self.valid_loader, ncols=200,
            desc=f"{epoch} EPOCH Validation", total=len(self.valid_loader)
        )
        for image, label_d, label_g, label_e in progress:
            image = image.to(self.device, dtype=torch.float)
            label_d = label_d.to(self.device, dtype=torch.float)
            label_g = label_g.to(self.device, dtype=torch.float)
            label_e = label_e.to(self.device, dtype=torch.float)

            logits = self.model(image)
            loss, (loss_d, loss_g, loss_e) = self.get_loss(
                logits, (label_d, label_g, label_e)
            )   

            losses += loss.item()
            losses_t[0] += loss_d.item()
            losses_t[1] += loss_g.item()
            losses_t[2] += loss_e.item()

            total += image.shape[0]
            crts = self.get_n_correct(logits, (label_d, label_g, label_e))
            crt_list[0] += crts[0]
            crt_list[1] += crts[1]
            crt_list[2] += crts[2]

        valid_loss = losses / len(self.valid_loader)
        valid_loss_t = [lst/len(self.valid_loader) for lst in losses_t]
        valid_acc_t = [c/total for c in crt_list]
        valid_acc = np.mean(valid_acc_t)
        return valid_loss, valid_acc, valid_loss_t, valid_acc_t

    def get_wce_info(self):
        nd = [593, 6975, 1074, 1686, 576, 130, 443]
        ng = [141, 937, 2826, 2759, 4291, 523]
        ne = [6285, 3413, 1779]
        wd = torch.Tensor([1-(d/sum(nd)) for d in nd]).to(self.device)
        wg = torch.Tensor([1-(g/sum(ng)) for g in ng]).to(self.device)
        we = torch.Tensor([1-(e/sum(ne)) for e in ne]).to(self.device)
        return (wd, wg, we)
        
    @torch.no_grad()
    def inference(self):
        model_path = Path(self.cfg.save_dir) / "model" / "model.pt"
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        raise NotImplementedError


if __name__ == "__main__":
    from utils import get_config
    from loader import add_fold, get_loader

    config = get_config()
    config.base.save_dir = False
    config.log.wandb = False
    config.train.n_epochs = 1

    data = add_fold(config.data.data_dir)
    loader = get_loader(data, config, None)
    solver = Solver(config, loader)
    solver.train()
