import cv2
import numpy as np
import pandas as pd

import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


def set_transforms(cfg: dict) -> alb.Compose:
    return {
        "train": alb.Compose([
            alb.HorizontalFlip(p=.4),
            alb.Blur(blur_limit=(1,3), p=.2),
            alb.CoarseDropout(max_holes=10, max_height=10, max_width=10, p=.2),
            alb.Downscale(scale_min=.8, scale_max=.9, p=.2),
            alb.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=.2
            ),
            alb.OpticalDistortion(distort_limit=.2, p=.2),
            alb.RandomBrightnessContrast(p=.2),
            alb.ShiftScaleRotate(rotate_limit=15, p=.2),
            # alb.Resize(input_size, input_size),
            alb.Normalize(),
            ToTensorV2()
        ], p=1.),
        "valid": alb.Compose([
            # alb.Resize(input_size, input_size),
            alb.Normalize(),
            ToTensorV2()
        ], p=1.)
    }


def get_loader(data:pd.DataFrame, cfg:dict, fold:int=None):
    if fold is None: fold = 0
    train = data[data.fold != fold].reset_index(drop=True)
    valid = data[data.fold == fold].reset_index(drop=True)

    transforms = set_transforms(cfg)

    train_loader = DataLoader(
        DatasetV1(train, cfg.image_size, transforms["train"]),
        batch_size=cfg.batch_size, num_workers=cfg.n_workers,
        shuffle=True, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        DatasetV1(valid, cfg.image_size, transforms["valid"]),
        batch_size=cfg.batch_size*2, num_workers=cfg.n_workers,
        shuffle=False, pin_memory=True, drop_last=False
    )
    return train_loader, valid_loader, (train, valid)


class DatasetV1(Dataset):
    def __init__(self, data, imgsz, transforms=None):
        super().__init__()
        self.data = data
        self.imgsz = imgsz
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.loc[idx]
        image = self.prep_image(data)
        return image, {
            "daily": data.Daily,
            "gender": data.Gender,
            "embel": data.Embellishment
        }

    def prep_image(self, data):
        img = cv2.cvtColor(
            cv2.imread(data.image_name), cv2.COLOR_BGR2RGB
        )
        img = self.crop_bbox(
            img, data.BBox_xmin, data.BBox_ymin, data.BBox_xmax, data.BBox_ymax
        )
        img = self.fill_bg(img, self.imgsz).astype(np.float32)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img

    @staticmethod
    def crop_bbox(img, xmin, ymin, xmax, ymax):
        return img[
            ymin: ymax, xmin: xmax, ...
        ]

    @staticmethod
    def fill_bg(img, imgsz):
        h, w = img.shape[:2]
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
