import os, random
from pathlib import Path

import pandas as pd
import numpy as np
from box import Box
from sklearn.model_selection import StratifiedKFold

import torch
import torch.optim as optim


def validate(cfg):
    assert isinstance(cfg.seed, int)
    assert isinstance(cfg.gpu_no, int)
    assert cfg.device in ["cuda", "cpu"]
    assert cfg.train.n_fold in [1, 4]


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def add_fold(cfg: Box, fold_path:str=None) -> pd.DataFrame:
    if fold_path is None:
        fold_path = "./train_fold.csv"

    if Path(fold_path).is_file():
        return pd.read_csv(fold_path)

    labels = ["Daily", "Gender", "Embellishment"]

    image_dir = os.path.join(cfg.data.data_dir, "train")
    data_path = os.path.join(cfg.data.data_dir, "info_etri20_emotion_train.csv")

    data = pd.read_csv(data_path)
    data["image_name"] = image_dir + "/" + data.image_name.map(str)
    data["uc"] = data[labels].apply(lambda x: f"{x[0]}{x[1]}{x[2]}", axis=1)
    data["fold"] = 0

    spliter = StratifiedKFold(n_splits=4, shuffle=True, random_state=cfg.seed)
    uc = data["uc"].values
    for fold, (_, idx) in enumerate(spliter.split(uc, uc)):
        data.loc[idx, "fold"] = fold

    data.to_csv(fold_path, index=False)
    return data



def set_optimizer(opt_config, model):
    opt_name = opt_config.type.lower()
    if opt_name == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=opt_config.learning_rate,
            betas=(opt_config.beta_1, opt_config.beta_2),
            weight_decay=opt_config.weight_decay
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=opt_config.learning_rate,
            betas=(opt_config.beta_1, opt_config.beta_2),
            weight_decay=opt_config.weight_decay
        )
    elif opt_name == "radam":
        optimizer = optim.RAdam(
            params=model.parameters(),
            lr=opt_config.learning_rate,
            betas=(opt_config.beta_1, opt_config.beta_2),
            weight_decay=opt_config.weight_decay
        )
    else:
        raise ValueError
    return optimizer


def set_scheduler(sch_config, opt_config, optimizer):
    sch_name = sch_config.type.lower()
    if sch_name == "CosineAnnealingLR".lower():
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=sch_config.t_max,
            eta_min=opt_config.min_learning_rate
        )
    elif sch_name == "CosineAnnealingWarmRestarts".lower():
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=sch_config.t_0,
            T_mult=sch_config.t_mult
            eta_min=opt_config.min_learning_rate
        )
    elif sch_name == "ReduceLROnPlateau".lower():
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=sch_config.factor,
            patience=sch_config.patience,
            min_lr=opt_config.min_learning_rate
        )
    elif sch_name == "LambdaLR".lower():
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda= lambda epoch: 0.9*epoch
        )
    elif sch_name == "LinearLR".lower():
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=sch_config.start_factor,
            total_iters=sch_config.total_iters
        )
    else:
        scheduler = None
    return scheduler
