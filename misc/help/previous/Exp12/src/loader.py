import cv2
import math
import pydicom
import numpy as np
from glob import glob
from os.path import join

from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 use_patch=False,
                 batch_size=4,
                 img_size=512,
                 augmentation=None,
                 **kwargs):
        super(DataLoader, self).__init__()
        assert mode in ["train", "valid"]

        data_dirs = glob(join(data_dir, "*"))
        input_paths = [glob(join(f, "*.dcm"))[0] for f in data_dirs]
        target_paths = [sorted(glob(join(f, "*.png"))) for f in data_dirs]

        valid_ratio = .1
        valid_length = int(len(input_paths) * valid_ratio)
        train_X_paths, valid_X_paths = input_paths[valid_length:], input_paths[:valid_length]
        train_y_paths, valid_y_paths = target_paths[valid_length:], target_paths[:valid_length]

        self.mode = mode
        if mode == "train":
            self.input_paths = train_X_paths
            self.target_paths = train_y_paths
        else:
            self.input_paths = valid_X_paths
            self.target_paths = valid_y_paths

        self.indexes = np.arange(len(self.input_paths))
        self.use_patch = use_patch

        if self.use_patch:
            self.batch_size = 1
            self.n_patches = batch_size
        else:
            self.batch_size = batch_size

        self.img_size = img_size
        self.augmentation = augmentation
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]

        if self.use_patch:
            bx, by = self.patch_prep(self.input_paths[indexes[0]], self.target_paths[indexes[0]])
            return bx, by
        else:
            bx, by = [], []
            for i in indexes:
                image = self.x_lr_prep(self.input_paths[i])
                mask = self.y_lr_prep(self.target_paths[i])
                if self.augmentation:
                    aug = self.augmentation(image=image, mask=mask)
                    image = normalize(aug["image"])[..., np.newaxis]
                    mask = normalize(aug["mask"])
                    bx.append(image)
                    by.append(mask)
                else:
                    bx.append(image[..., np.newaxis])
                    by.append(mask)
            return np.array(bx), np.array(by)

    def x_lr_prep(self, path):
        img = pydicom.dcmread(path).pixel_array
        # img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        img = random_histeq(img)
        return normalize(cv2.resize(img, (self.img_size, self.img_size))).astype(np.float32)

    def y_lr_prep(self, path):
        img = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in path]
        img = normalize(np.stack(img, axis=-1)).astype(np.float32)
        bg = 1 - img.sum(axis=-1, keepdims=True)
        bg = clip_0_1(bg)
        img = np.concatenate((img, bg), axis=-1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img[img>0] = 1.
        return img

    def patch_prep(self, input_path, target_path):
        img_x = pydicom.dcmread(input_path).pixel_array
        img_x = normalize(img_x).astype(np.float32)[..., np.newaxis]
        img_y = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in target_path]
        img_y = normalize(np.stack(img_y, axis=-1)).astype(np.float32)
        bg_y = 1 - img_y.sum(axis=-1, keepdims=True)
        bg_y = clip_0_1(bg_y)
        img_y = np.concatenate((img_y, bg_y), axis=-1)

        bx, by, inr = [], [], 0
        while inr < (self.n_patches // 2):
            px, py = self.random_crop(img_x, img_y)
            if not np.all(py[...,:8] == 0):
                if self.augmentation:
                    aug = self.augmentation(image=px, mask=py)
                    image = normalize(aug["image"])
                    mask = normalize(aug["mask"])
                    bx.append(image)
                    by.append(mask)
                else:
                    bx.append(px)
                    by.append(py)
                inr += 1

        for i in range(inr, self.n_patches):
            px, py = self.random_crop(img_x, img_y)
            bx.append(px)
            by.append(py)

        return np.array(bx), np.array(by)

    def random_crop(self, x, y):
        h, w, _ = x.shape
        i = np.random.randint(0, (h - self.img_size))
        j = np.random.randint(0, (w - self.img_size))
        return x[i:(i + self.img_size), j:(j + self.img_size)], y[i:(i + self.img_size), j:(j + self.img_size), :]


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def clip_0_1(x):
    x[x < 0] = 0
    return x


def random_histeq(x):
    if np.random.random() < .5:
        g = np.random.randint(1, 10)
        return cv2.createCLAHE(tileGridSize=(g, g)).apply(x)
    else:
        return x


# def rotate_90(img, mask):
#     if np.random.random() < .5:
#         img = img[...,0].T
#         mask = mask[...,0].T
#         return img, mask
#     else:
#         return img, mask
