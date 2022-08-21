import os, time
import argparse
from pathlib import Path
from shutil import copyfile

import yaml
import torch
from box import Box
from utils import validate, fix_seed, add_fold
from loader import get_loader
from solver import Solver


PFX = "***** "

def main(cfg):
    # validate(cfg)
    fix_seed(cfg.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_number)

    if torch.cuda.is_available():
        print(f"{PFX}Use {torch.cuda.get_device_name()}")

    save_dir = Path(cfg.save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
        Path(save_dir / "model").mkdir()
        print(f"{PFX}Make {save_dir.as_posix()}")
        time.sleep(1)

    copyfile("./config.yaml", save_dir / "config.yaml")
    data = add_fold(cfg)

    if cfg.train.n_fold == 1:
        print(f"{PFX}HOLDOUT")
        train_loader, valid_loader, _ = get_loader(data, cfg.data, None)
        solver = Solver(cfg, (train_loader, valid_loader), None)
        best = solver.train()
        print(f"{PFX}Best Score: {best}")

    # else:
    #     bests = []
    #     for fold in range(1, cfg.train.n_fold+1):
    #         print(f"{PFX}FOLD {fold}")
    #         train_loader, valid_loader = get_loader(fold)
    #         model = get_model()
    #         solver = Solver()
    #         bests.append(solver.train())

    #     print(f"{PFX}bests")
    #     print(f"{PFX}Best Avg Score: {sum(bests)/fold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader))

    print(config)
    main(config)
