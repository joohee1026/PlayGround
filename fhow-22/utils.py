import os, random

import yaml
import numpy as np
from box import Box

import torch
import torch.nn as nn
from torch.optim import (
    SGD, RMSprop, Adam, AdamW, NAdam, RAdam
)
from torch.optim.lr_scheduler import (
    LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, 
    # ConstantLR, LinearLR, CyclicLR
)


def get_config(config_path:str="config.yaml") -> Box:
    with open(config_path) as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader))
        config["base"]["config_path"] = config_path
    
    return config


def fix_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Validator:
    def __init__(self, config:Box):
        assert isinstance(config, Box)
        self.config = config

    def main(self):
        assert isinstance(self.config.base.seed, int)
        assert isinstance(self.config.base.gpu_number, int)
        assert self.config.base.device in ("cuda", "cpu")

    def optimizer(self):
        assert "optimizer" not in self.config.keys()
        assert self.config.optimizer.name.lower() in \
            ("sgd", "rmsprop", "adam", "adamw")

    def scheduler(self):
        assert "scheduler" not in self.config.keys()
        assert self.config.scheduler.name.lower() in \
            ("lambdalr", "reducelronplateau", "cosineannealinglr",
             "cosineannealingwarmrestarts", "steplr", "cycliclr",)


class SetOptimizer:
    def __init__(self, config:Box, model:nn.Module):
        Validator(config).optimizer
        self.optimizers = {
            "sgd": self.sgd,
            "rmsprop": self.rmsprop,
            "adam": self.adam,
            "adamw": self.adamw,
            "nadam": self.nadam,
            "radam": self.radam
        }
        self.model_params = model.parameters()
        self.load = self.optimizers[config.name.lower()](**config)

    def sgd(self, lr=1e-4, momentum=0, weight_decay=0, nesterov=False, **kwargs):
        return SGD(
            params=self.model_params,
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )

    def rmsprop(self, lr=1e-4, momentum=0, alpha=.99, weight_decay=0, **kwargs):
        return RMSprop(
            params=self.model_params,
            lr=lr, momentum=momentum, alpha=alpha, weight_decay=weight_decay
        )

    def adam(self, lr=1e-4, betas=(.9,.999), weight_decay=0, amsgrad=False, **kwargs):
        return Adam(
            params=self.model_params,
            lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad
        )

    def adamw(self, lr=1e-4, betas=(.9,.999), weight_decay=0, amsgrad=False, **kwargs):
        return AdamW(
            params=self.model_params, 
            lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad
        )
    
    def nadam(self, lr=1e-4, betas=(.9,.999), weight_decay=0, momentum_decay=4e-3, **kwargs):
        return NAdam(
            params=self.model_params,
            lr=lr, betas=betas, weight_decay=weight_decay, momentum_decay=momentum_decay
        )

    def radam(self, lr=1e-4, betas=(.9,.999), weight_decay=0, **kwargs):
        return RAdam(
            params=self.model_params,
            lr=lr, betas=betas, weight_decay=weight_decay
        )
    

class SetScheduler:
    def __init__(self, config:Box, optimizer:torch.optim, lambda_func=None):
        Validator(config).scheduler
        self.schedules = {
            "lambdalr": self.lambdalr,
            "reducelronplateau": self.reducelronplateau,
            "cosineannealinglr": self.cosineannealinglr,
            "cosineannealingwarmrestarts": self.cosineannealingwarmrestarts,
            "steplr": self.steplr,
            # "cycliclr": self.cycliclr,
            # "constantlr": self.constantlr,
            # "linearlr": self.linearlr
        }
        self.optimizer = optimizer
        self.lambda_func = lambda_func
        self.load = self.schedules[config.name.lower()](**config)

    def lambdalr(self, last_epoch=-1, **kwargs):
        return LambdaLR(
            optimizer=self.optimizer, lr_lambda=self.lambda_func, last_epoch=last_epoch
        )

    def reducelronplateau(self, mode="min", factor=.1, patience=10, min_lr=0, **kwargs):
        return ReduceLROnPlateau(
            optimizer=self.optimizer, 
            mode=mode, factor=factor, patience=patience, min_lr=min_lr
        )

    def cosineannealinglr(self, t_max=100, min_lr=0, last_epoch=-1, **kwargs):
        return CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=t_max, eta_min=min_lr, last_epoch=last_epoch
        )
    
    def cosineannealingwarmrestarts(self, t_0=100, t_mult=1, min_lr=0, last_epoch=-1, **kwargs):
        return CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=t_0, T_mult=t_mult, eta_min=min_lr, last_epoch=last_epoch
        )

    def steplr(self, step_size=10, gamma=.1, last_epoch=-1, **kwargs):
        return StepLR(
            optimizer=self.optimizer, 
            step_size=step_size, gamma=gamma, last_epoch=last_epoch
        )

    def cycliclr(self):
        raise NotImplementedError

    def constantlr(self):
        raise NotImplementedError

    def linearlr(self):
        raise NotImplementedError
