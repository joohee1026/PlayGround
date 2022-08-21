import os
import time

import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_

from model import get_model
from utils import set_optimizer, set_scheduler


class Solver:
    def __init__(self, cfg, loader, fold=None):
        self.cfg = cfg
        self.train_loader, self.valid_loader = loader

        self.device = cfg.device

        self.n_epoch = cfg.train.n_epoch

        self.model = get_model(cfg.train.model_name)
        self.model.to(self.device)

        self.amp = cfg.train.amp
        self.clip = cfg.train.max_grad_norm
        self.prefix = f"FOLD{fold}" if isinstance(fold, int) else "HOLDOUT"

        if cfg.loss.wce:
            raise NotImplementedError
        else:
            self.criterions = [nn.CrossEntropyLoss() for _ in range(3)]
        self.wd, self.wg, self.we = cfg.loss.task_weight

        self.optimizer = set_optimizer(cfg.optimizer, self.model)
        self.scheduler = set_scheduler(cfg.scheduler, cfg.optimizer, self.optimizer)

        if cfg.log.is_log:
            wandb.init(
                project=cfg.log.project,
                entity=cfg.log.entity,
                # group=cfg.log.group_name,
                name=cfg.log.name,
                config=dict(cfg)
            )

    def train(self):
        start = time.time()
        best_acc = 0.
        for epoch in range(1, self.n_epoch+1):
            trn_loss, trn_acc, trn_loss_t, trn_acc_t  = self.update(epoch)
            val_loss, val_acc, val_loss_t, val_acc_t = self.validation(epoch)

            if self.cfg.scheduler.type in ("ReduceLROnPlateau", "LinearLR"):
                self.scheduler.step(val_loss)

            if val_acc > best_acc:
                print(f"Improved {best_acc:.3f} >> {val_acc:.3f}")
                best_acc = val_acc
                model_dir = os.path.join(self.cfg.save_dir, "model")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(model_dir, "model.pt")
                )

            elapsed = time.time() - start
            print(f"EPOCH{epoch} complete in {elapsed:.0f}s")
            print(f"TRAIN LOSS: {trn_loss:.3f} / ACC: {trn_acc:.3f}")
            print(f"VALID LOSS: {val_loss:.3f} / ACC: {val_acc:.3f}")

            if self.cfg.log.is_log:
                lr = self.optimizer.param_groups[0]["lr"]
                wandb.log({
                    "Train Loss": trn_loss,
                    "Train Loss D": trn_loss_t[0],
                    "Train Loss G": trn_loss_t[1],
                    "Train Loss E": trn_loss_t[2],
                    "Train Acc": trn_acc,
                    "Train Acc D": trn_acc_t[0],
                    "Train Acc G": trn_acc_t[1],
                    "Train Acc E": trn_acc_t[2],
                    "Valid Loss": val_loss,
                    "Valid Loss D": val_loss_t[0],
                    "Valid Loss G": val_loss_t[1],
                    "Valid Loss E": val_loss_t[2],
                    "Valid Acc": val_acc,
                    "Valid Acc D": val_acc_t[0],
                    "Valid Acc G": val_acc_t[1],
                    "Valid Acc E": val_acc_t[2],
                    "Learning Rate": lr
                })

        elapsed = time.time() - start
        print(f"Complete in {elapsed:.0f}s")

    def get_loss(self, logits, labels):
        losses = [
            self.criterions[i](logits[i], labels[i]) for i in range(3)
        ]
        return (
            self.wd*losses[0] + self.wg*losses[1] + self.we*losses[2],
            losses
        )

    def get_n_correct(self, logits, labels):
        predictions = [torch.max(logit, axis=1)[1] for logit in logits]
        crts = [
            (pred == label).sum().item() for pred, label in zip(predictions, labels)
        ]
        return crts

    def update(self, epoch):
        self.model.train()

        losses, losses_t = 0., [0. for _ in range(3)]
        total, crt_t = 0, [0 for _ in range(3)]
        progress = tqdm(
            enumerate(self.train_loader), ncols=200,
            desc=f"[{self.prefix}]-[EPOCH {epoch}/{self.n_epoch}]",
            total=len(self.train_loader)
        )
        for i, (image, labels) in progress:
            image = image.to(self.device, dtype=torch.float)
            label_d = labels["daily"].to(self.device, dtype=torch.long)
            label_g = labels["gender"].to(self.device, dtype=torch.long)
            label_e = labels["embel"].to(self.device, dtype=torch.long)

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
                self.cfg.scheduler.type not in ("ReduceLROnPlateau", "LinearLR"):
                self.scheduler.step()

            losses += loss.item()
            losses_t[0] += loss_d.item()
            losses_t[1] += loss_g.item()
            losses_t[2] += loss_e.item()
            train_loss = losses / (i+1)
            train_loss_t = [lst/(i+1) for lst in losses_t]

            total += image.shape[0]
            crts = self.get_n_correct(logits, (label_d, label_g, label_e))
            crt_t[0] += crts[0]
            crt_t[1] += crts[1]
            crt_t[2] += crts[2]
            train_acc_t = [c/total for c in crt_t]
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
        total, crt_t = 0, [0 for _ in range(3)]
        progress = tqdm(
            enumerate(self.valid_loader), ncols=200,
            desc=f"{epoch}EPOCH Validation", total=len(self.valid_loader)
        )
        for i, (image, labels) in progress:
            image = image.to(self.device, dtype=torch.float)
            label_d = labels["daily"].to(self.device, dtype=torch.long)
            label_g = labels["gender"].to(self.device, dtype=torch.long)
            label_e = labels["embel"].to(self.device, dtype=torch.long)


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
            crt_t[0] += crts[0]
            crt_t[1] += crts[1]
            crt_t[2] += crts[2]

        valid_loss = losses / (i+1)
        valid_loss_t = [lst/(i+1) for lst in losses_t]
        valid_acc_t = [c/total for c in crt_t]
        valid_acc = np.mean(valid_acc_t)
        return valid_loss, valid_acc, valid_loss_t, valid_acc_t


if __name__ == "__main__":
    from loader import get_loader
