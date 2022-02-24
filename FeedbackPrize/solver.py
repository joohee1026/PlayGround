import os, gc
import time
import wandb
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.cuda import amp

from utils import *
from loss import DiceLoss
from loader import ID2LABEL


class Solver:
    def __init__(self, cfg, model, train_loader, valid_loader=None, fold=None):
        self.cfg = cfg
        self.sch_cfg = cfg.scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.fold = fold
        self.prefix = f"FOLD{fold}" if isinstance(fold, int) else "HOLDOUT"

        self.device = cfg.device
        self.n_epoch = cfg.train.n_epoch
        self.n_fold = cfg.train.n_fold
        self.n_batch = cfg.train.n_batch

        self.model = model
        self.model.to(self.device)
        self.n_classes = len(ID2LABEL)

        self.amp = cfg.train.amp
        self.clip = cfg.train.max_grad_norm

        if cfg.loss.wce:
            cweight = torch.tensor(cfg.loss.wce_w, dtype=torch.float32).to(self.device)
            self.cce_fn = nn.CrossEntropyLoss(weight=cweight)
        else:
            self.cce_fn = nn.CrossEntropyLoss()

        self.use_dice = False
        if cfg.loss.dice > 0:
            self.use_dice = True
            self.dice_fn = DiceLoss(alpha=.01, reduction="mean", with_logits=False)
            self.ccew, self.dicew = cfg.loss.cce, cfg.loss.dice
        self.optimizer = set_optimizer(cfg.optimizer, self.model)
        self.scheduler = set_scheduler(cfg.scheduler, cfg.optimizer, self.optimizer)

        if cfg.log.logging:
            wandb.init(
                project=cfg.log.project,
                entity=cfg.log.entity,
                group=cfg.log.group_name,
                name=cfg.log.name,
                config=dict(cfg)
            )
            # wandb.watch(self.model)

    def train(self):
        start = time.time()
        best_f1 = 0.
        for epoch in range(1, self.n_epoch+1):
            gc.collect()
            train_loss, train_acc, train_cce, train_dice = self.train_step(epoch)
            valid_loss, valid_acc, valid_cce, valid_dice, valid_f1 = self.validation()
            mean_f1 = np.mean(list(valid_f1.values()))

            if self.cfg.log.logging:
                lr = self.optimizer.param_groups[0]["lr"]
                train_log = {
                    "Train Epoch Loss": train_loss,
                    "Train Epoch Accuracy": train_acc,
                    "Valid Epoch Loss": valid_loss,
                    "Valid Epoch Accuracy": valid_acc,
                    "Valid Epoch F1": mean_f1,
                    "Learning Rate": lr
                }
                for k, v in valid_f1.items():
                    train_log[k] = v

                if self.use_dice:
                    train_log["Train Epoch CCE"] = train_cce
                    train_log["Train Epoch DICE"] = train_dice
                    train_log["Valid Epoch CCE"] = valid_cce
                    train_log["Valid Epoch DICE"] = valid_dice

                wandb.log(train_log)

            if self.sch_cfg.type in ["ReduceLROnPlateau", "LinearLR"]:
                self.scheduler.step(valid_loss)

            prefix = f"[{self.prefix}]-[EPOCH {epoch}/{self.n_epoch}]"
            self.on_epoch_end(prefix, valid_loss, valid_f1)

            if best_f1 < mean_f1:
                print(f"{prefix} ** => [Improved {best_f1:.4f} >> {mean_f1:.4f}]")
                best_f1 = mean_f1
                save_path = os.path.join(self.cfg.save_dir, f"{self.prefix.lower()}.pt")
                torch.save(self.model.state_dict(), save_path)

        if self.cfg.log.logging:
            wandb.config.update({f"{self.prefix} F1": best_f1})
            if not self.cfg.log.fold_cont:
                wandb.finish()

        end = time.time() - start
        print(f"===== Complete in {end:.0f}s, {self.prefix} best F1: {best_f1:.4f}")
        return best_f1

    def train_step(self, epoch):
        self.model.train()
        scaler = amp.GradScaler() if self.amp else None

        losses, accs = 0., 0.
        cce, dice = 0., 0.
        progress = tqdm(
            enumerate(self.train_loader), ncols=150,
            desc=f"[{self.prefix}]-[EPOCH {epoch}/{self.n_epoch}]",
            total=len(self.train_loader)
        )
        for i, data in progress:
            ids = data["input_ids"].to(self.device, dtype=torch.long)
            mask = data["attention_mask"].to(self.device, dtype=torch.long)
            labels = data["labels"].to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            if self.amp:
                with amp.autocast(enabled=True):
                    logits = self.model(
                        input_ids=ids, attention_mask=mask, return_dict=False
                    )[0]
                    cce_loss = self.cce_fn(logits.view(-1, self.n_classes), labels.view(-1))
                    if self.use_dice:
                        dice_loss = self.dice_fn(logits, labels)
                        loss = self.ccew * cce_loss + self.dicew * dice_loss
                    else:
                        loss = cce_loss
                scaler.scale(loss).backward()
                if self.clip:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                logits = self.model(
                    input_ids=ids, attention_mask=mask, return_dict=False
                )[0]
                cce_loss = self.cce_fn(logits.view(-1, self.n_classes), labels.view(-1))
                if self.use_dice:
                    dice_loss = self.dice_fn(logits, labels)
                    loss = self.ccew * cce_loss + self.dicew * dice_loss
                else:
                    loss = cce_loss
                loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()

            if (self.scheduler is not None) and \
                    (self.sch_cfg.type not in ["ReduceLROnPlateau", "LinearLR"]):
                self.scheduler.step()

            losses += (loss.item() * self.n_batch)
            train_loss = losses / ((i+1) * self.n_batch)

            if self.use_dice:
                cce += cce_loss.item()
                train_cce = cce / (i+1)
                dice += dice_loss.item()
                train_dice = dice / (i+1)
            else:
                train_cce, train_dice = train_loss, 0

            accs += self.get_acc(labels, logits)
            train_acc = accs / (i+1)

            lr = self.optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                LR=lr, TOTAL_LOSS=train_loss, ACC=train_acc, CCE=train_cce, DICE=train_dice
            )

        gc.collect()
        return train_loss, train_acc, train_cce, train_dice

    def get_acc(self, labels, logits):
        mask = labels.view(-1) != -100
        flat_true = labels.view(-1)
        masked_true = torch.masked_select(flat_true, mask)

        flat_pred = torch.argmax(logits.view(-1, self.model.num_labels), axis=1)
        masked_pred = torch.masked_select(flat_pred, mask)
        return accuracy_score(masked_true.cpu().numpy(), masked_pred.cpu().numpy())

    def on_epoch_end(self, prefix, loss, f1):
        print(f"{prefix}")
        print(f"{prefix} ** [Validation Loss]")
        print(f"{prefix} ** : {loss:.4f}")
        print(f"{prefix}")
        print(f"{prefix} ** [Validation F1]")
        for k, v in f1.items():
            print(f"{prefix} ** {k:<10}: {v:.4f}")
        print(f"{prefix}")
        print(f"{prefix} ** => Overall F1: {np.mean(list(f1.values())):.4f}")

    def set_validations(self, raw, data):
        self.raw = raw
        self.valid_data = data

    def validation(self):
        _df_valid = self.raw.loc[self.raw["id"].isin(self.valid_data.id)]
        all_labels, valid_loss, valid_acc, valid_cce, valid_dice = self.inference()
        oof = self.get_predictions(all_labels)

        if len(oof) == 0:
            f1_result = {"None": 0}
        else:
            f1_result = {}
            for cls in oof["class"].unique():
                pdf = oof.loc[oof["class"] == cls].copy()
                gdf = _df_valid.loc[_df_valid["discourse_type"] == cls].copy()
                f1 = score_feedback_comp(pdf, gdf)
                f1_result[f"F1-{cls}"] = f1
        return valid_loss, valid_acc, valid_cce, valid_dice, f1_result

    @torch.no_grad()
    def inference(self):
        self.model.eval()

        losses, accs = 0., 0.
        cce, dice = 0., 0.
        predictions = defaultdict(list)
        seen_words_idx = defaultdict(list)
        progress = tqdm(
            enumerate(self.valid_loader), ncols=50,
            desc="Validation", total=len(self.valid_loader)
        )
        for i, batch in progress:
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = self.model(
                ids, attention_mask=mask, return_dict=False
            )[0]
            cce_loss = self.cce_fn(logits.view(-1, self.n_classes), labels.view(-1))
            if self.use_dice:
                dice_loss = self.dice_fn(logits, labels)
                loss = self.ccew * cce_loss + self.dicew * dice_loss
            else:
                loss = cce_loss

            losses += (loss.item() * (self.n_batch*2))
            valid_loss = losses / ((i+1) * (self.n_batch*2))

            if self.use_dice:
                cce += cce_loss.item()
                valid_cce = cce / (i+1)
                dice += dice_loss.item()
                valid_dice = dice / (i+1)
            else:
                valid_cce, valid_dice = valid_loss, 0

            accs += self.get_acc(labels, logits)
            valid_acc = accs / (i+1)

            batch_preds = torch.argmax(logits, axis=-1).cpu().numpy()
            for k, (chunk_preds, text_id) in enumerate(zip(batch_preds, batch['overflow_to_sample_mapping'].tolist())):
                word_ids = batch['word_ids'][k].numpy()
                chunk_preds = [ID2LABEL[i] for i in chunk_preds]
                for idx, word_idx in enumerate(word_ids):
                    if word_idx == -1:
                        pass
                    elif word_idx not in seen_words_idx[text_id]:
                        predictions[text_id].append(chunk_preds[idx])
                        seen_words_idx[text_id].append(word_idx)

        final_predictions = [predictions[k] for k in sorted(predictions.keys())]
        return final_predictions, valid_loss, valid_acc, valid_cce, valid_dice

    def get_predictions(self, all_labels):
        final_preds = []
        for i in range(len(self.valid_data)):
            idx = self.valid_data.id.values[i]
            pred = all_labels[i]
            j = 0
            while j < len(pred):
                cls = pred[j]
                if cls == 'O': pass
                else: cls = cls.replace('B','I')
                end = j + 1
                while end < len(pred) and pred[end] == cls:
                    end += 1
                if cls != 'O' and cls != '' and end - j > 7:
                    final_preds.append((idx, cls.replace('I-',''),
                                ' '.join(map(str, list(range(j, end))))))
                j = end

        if len(final_preds) == 0:
            df_pred = pd.DataFrame([], columns = ['id','class','predictionstring'])
        else:
            df_pred = pd.DataFrame(final_preds)
            df_pred.columns = ['id','class','predictionstring']
        return df_pred
