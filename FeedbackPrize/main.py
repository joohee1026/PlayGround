import os, time
import argparse
import yaml
from box import Box
from shutil import copyfile

import torch
from utils import fix_seed, confirm_params
from loader import get_data, get_loader
from model import save_model_info, get_model
from solver import Solver


def main(cfg):
    confirm_params(cfg)
    fix_seed(cfg.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_no)

    if torch.cuda.is_available():
        print(f"** Using GPU: {torch.cuda.get_device_name()}")

    save_dir = cfg.save_dir
    model_dir = os.path.join(cfg.save_dir, "model")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(model_dir)
        print("** Build Directories")

    time.sleep(3)
    copyfile("./config.yaml", os.path.join(cfg.save_dir, "config.yaml"))

    raw, _, data = get_data(cfg.data.data_dir)
    save_model_info(cfg.model.name, model_dir)
    print("** Init Model Prepared")

    if cfg.train.n_fold == 1:
        train_loader, valid_loader, prep = get_loader(
            data, model_dir, cfg.train.n_batch,
            cfg.data.max_length, cfg.data.stride, "holdout"
        )
        model = get_model(model_dir)
        solver = Solver(cfg, model, train_loader, valid_loader, "holdout")
        solver.set_validations(raw, prep[1])
        best = solver.train()
        print(f"** Best Overall F1 Score: {best}")

    else:
        best_log = []
        for fold in range(1, cfg.train.n_fold+1):
            print(f"### FOLD {fold} ###")
            train_loader, valid_loader, prep = get_loader (
                data, model_dir, cfg.train.n_batch,
                cfg.data.max_length, cfg.data.stride, fold, cfg.train.n_fold
            )
            model = get_model(model_dir)
            solver = Solver(cfg, model, train_loader, valid_loader, fold)
            solver.set_validations(raw, prep[1])
            best = solver.train()
            best_log.append(best)

        print(best_log)
        print(f">>> avg: {sum(best_log)/cfg.train.n_fold}")


if __name__ == "__main__":
    # os.system("wandb login --relogin")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./config.yaml")
    args = parser.parse_args()

    with open(args.data) as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader))

    main(config)
