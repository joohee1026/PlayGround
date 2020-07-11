import cv2
import math
import pydicom
import numpy as np
from glob import glob
from os.path import join

from tensorflow.keras.utils import Sequence


class DataLoader_lr(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 batch_size=4,
                 img_size=512,
                 augmentation=None,
                 **kwargs):
        super(DataLoader_lr, self).__init__()
        assert mode in ["train", "valid"]

        data_dirs = glob(join(data_dir, "*"))
        input_paths = [glob(join(f, "*.dcm"))[0] for f in data_dirs]
        target_paths = [sorted(glob(join(f, "*.png"))) for f in data_dirs]

        train_X_paths, valid_X_paths = input_paths[20:], input_paths[:20]
        train_y_paths, valid_y_paths = target_paths[20:], target_paths[:20]

        self.mode = mode
        if mode == "train":
            self.input_paths = train_X_paths
            self.target_paths = train_y_paths
        else:
            self.input_paths = valid_X_paths
            self.target_paths = valid_y_paths

        self.indexes = np.arange(len(self.input_paths))
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

        bx, by = [], []
        for i in indexes:
            image = self.img_prep(self.input_paths[i])
            mask = self.mask_prep(self.target_paths[i])
            if self.augmentation:
                aug = self.augmentation(image=image, mask=mask)
                bx.append(aug["image"])
                by.append(aug["mask"])
            else:
                bx.append(image)
                by.append(mask)
        return np.array(bx)[..., np.newaxis], np.array(by)

    def img_prep(self, path):
        img = pydicom.dcmread(path).pixel_array
        return normalize(cv2.resize(img, (self.img_size, self.img_size))).astype(np.float32)

    def mask_prep(self, path):
        img = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in path]
        img = normalize(np.stack(img, axis=-1)).astype(np.float32)
        bg = 1 - img.sum(axis=-1, keepdims=True)
        img = np.concatenate((img, bg), axis=-1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img[img>0] = 1.
        return img


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def random_crop(x, y, crop_size, **kwargs):
    h, w = x.shape
    i = np.random.randint(0, (h-crop_size))
    j = np.random.randint(0, (w-crop_size))
    return x[i:(i+crop_size), j:(j+crop_size)], y[i:(i+crop_size), j:(j+crop_size), :]
