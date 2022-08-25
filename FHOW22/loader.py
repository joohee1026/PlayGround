import os, random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


N_CLASSES_TYPE = [7, 6, 3]


def add_fold(data_dir:str, fold_path:str=None, mode:str="train", seed:int=2022) -> pd.DataFrame:
    if fold_path is None:
        fold_path = f"./{mode}_fold.csv"

    if Path(fold_path).is_file():
        return pd.read_csv(fold_path)
    
    labels = ["Daily", "Gender", "Embellishment"]

    image_dir = os.path.join(data_dir, mode) 
    data_path = os.path.join(data_dir, f"info_etri20_emotion_{mode}.csv")

    data = pd.read_csv(data_path)
    data["image_name"] = image_dir + "/" + data.image_name.map(str)
    data["uc"] = data[labels].apply(lambda x: f"{x[0]}{x[1]}{x[2]}", axis=1)
    data["fold"] = 0

    spliter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    uc = data["uc"].values
    for fold, (_, idx) in enumerate(spliter.split(uc, uc)):
        data.loc[idx, "fold"] = fold
    
    data.to_csv(fold_path, index=False)
    return data


def set_transforms(cfg: dict) -> A.Compose:
    return {
        "train": A.Compose([
            A.HorizontalFlip(p=.4),
            A.Blur(blur_limit=(1,3), p=.2),
            A.CoarseDropout(max_holes=10, max_height=10, max_width=10, p=.2),
            A.Downscale(scale_min=.8, scale_max=.9, p=.2),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=.2
            ),
            A.OpticalDistortion(distort_limit=.2, p=.2),
            A.RandomBrightnessContrast(p=.2),
            A.ShiftScaleRotate(rotate_limit=15, p=.2),
            # alb.Resize(input_size, input_size),
            A.Normalize(),
            ToTensorV2()     
        ], p=1.),
        "valid": A.Compose([
            # alb.Resize(input_size, input_size),
            A.Normalize(),
            ToTensorV2()
        ], p=1.)
    }


def get_loader(data:pd.DataFrame, cfg:dict, fold:int=None) -> DataLoader:
    if fold is None: 
        fold = 0
    train = data[data.fold != fold].reset_index(drop=True)
    valid = data[data.fold == fold].reset_index(drop=True)

    imgsz = cfg.data.image_size
    cutmix_prob = cfg.train.cutmix_prob
    batch_size = cfg.train.batch_size
    n_workers = cfg.train.n_workers

    transforms = set_transforms(cfg)

    train_loader = DataLoader(
        DatasetV1(train, imgsz, cutmix_prob, transforms["train"]), 
        batch_size=batch_size, num_workers=n_workers,
        shuffle=True, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        DatasetV1(valid, imgsz, 0, transforms["valid"]),
        batch_size=batch_size*2, num_workers=n_workers,
        shuffle=False, pin_memory=True, drop_last=False
    )
    return train_loader, valid_loader, (train, valid)


class DatasetV1(Dataset):
    def __init__(self, data:pd.DataFrame, imgsz:int, cutmix_prob:float, transforms:A.Compose=None):
        super().__init__()
        self.data = data
        self.imgsz = imgsz
        self.cutmix_prob = cutmix_prob
        self.transforms = transforms
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.loc[idx]
        image = self.prep_image(data)
        label = [
            data.Daily, data.Gender, data.Embellishment
        ]
        if np.random.random() < self.cutmix_prob:
            image, label = self.apply_cutmix(image, label)
        else:
            label = [np.eye(n)[c] for n, c in zip(N_CLASSES_TYPE, label)]
            
        label = [lbl.astype(np.float32) for lbl in label]

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, label[0], label[1], label[2]

    def prep_image(self, data):
        img = cv2.cvtColor(
            cv2.imread(data.image_name), cv2.COLOR_BGR2RGB
        )
        img = self.crop_bbox(
            img, data.BBox_xmin, data.BBox_ymin, data.BBox_xmax, data.BBox_ymax
        )
        img = self.fill_bg(img, self.imgsz).astype(np.float32)
        return img

    def apply_cutmix(self, image, label):
        cut_size = image.shape[0] // 2
        label = [np.eye(n)[c] for n, c in zip(N_CLASSES_TYPE, label)]

        ref_idx = random.choice(range(len(self.data)))
        ref_data = self.data.loc[ref_idx]

        ref_image = self.prep_image(ref_data)
        ref_label = [np.eye(n)[c] for n, c in \
            zip(N_CLASSES_TYPE ,
            [ref_data.Daily, ref_data.Gender, ref_data.Embellishment])
        ]
        mixed_image = np.concatenate([
            image[:, :cut_size, :], ref_image[:, cut_size:, :]
        ], axis=1)
        mixed_label = [(i+j)/2 for i, j in zip(label, ref_label)]
        return mixed_image, mixed_label

    @staticmethod
    def crop_bbox(img, xmin, ymin, xmax, ymax):
        return img[
            ymin: ymax, xmin: xmax, ...
        ]

    @staticmethod
    def fill_bg(img, imgsz):
        w, h = img.shape[:2]
        hf = imgsz // 2
        if isinstance(imgsz, int):
            if h > w:
                nh, nw = imgsz, imgsz * w / h
            else:
                nh, nw = imgsz * h / w, imgsz
        else:
            nh, nw = imgsz

        nh, nw = int(nh), int(nw)
        img = cv2.resize(img, (nh, nw), interpolation=cv2.INTER_AREA)

        filled_img = np.zeros((imgsz, imgsz, 3))
        if h > w:
            filled_img[(hf-nw//2):(hf-nw//2+nw), ...] = img
        else:
            filled_img[:, (hf-nh//2):(hf-nh//2+nh), :] = img
        return filled_img


if __name__ == "__main__":
    from utils import get_config

    config = get_config()

    data = add_fold(config.data.data_dir)
    dataset = DatasetV1(
        data, config.data.image_size, config.train.cutmix_prob, None
    )
    image, label1, label2, label3 = dataset.__getitem__(0)
    print(image.shape, label1, label2, label3)

    loaders = get_loader(data, config, None)
    for loader in loaders[:-1]:
        for d in loader: break
        print(d[0].shape, d[1].shape, d[2].shape, d[3].shape)
            
