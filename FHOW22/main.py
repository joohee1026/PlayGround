import os, time
import argparse
from pathlib import Path
from shutil import copyfile

import wandb
import torch
from box import Box
from loguru import logger

from solver import Solver
from loader import add_fold, get_loader
from utils import Validator, fix_seed, get_config


def main(cfg:Box):
    logger.add(f"{cfg.base.save_dir}/log")
    logger.info(f"Config : {dict(cfg)}")

    Validator(cfg).main()
    fix_seed(cfg.base.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.base.gpu_number)
    if torch.cuda.is_available():
        logger.info(f"Use {torch.cuda.get_device_name()}")
    
    save_dir = Path(cfg.base.save_dir)
    model_dir = Path(save_dir) / "model"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Build {save_dir.as_posix()} directory")
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Build {model_dir.as_posix()} directory")
    time.sleep(1)

    copyfile("./config.yaml", save_dir / "config.yaml")
    logger.info(f"Copy config file")

    data = add_fold(cfg.data.data_dir)
    
    if cfg.train.n_fold == 1:
        logger.info(f"HOLDOUT train start")
        train_loader, valid_loader, _ = get_loader(data, cfg, None)
        solver = Solver(cfg, (train_loader, valid_loader), None)
        solver.train()
        
    # else:
    #     bests = []
    #     for fold in range(1, cfg.train.n_fold+1):
    #         train_loader, valid_loader = get_loader(fold)
    #         model = get_model()
    #         solver = Solver()
    #         bests.append(solver.train())
    # 
    # test_loader = get_loader()
    # solver = Solver(cfg, test_loader)
    # solver.inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)

    if config.log.wandb:
        wandb.init(
            project=config.log.project,
            entity=config.log.entity,
            name=config.log.name,
            config=dict(config)
        )
        main(config)
    else:
        main(config)
