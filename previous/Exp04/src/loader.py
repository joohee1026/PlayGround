import cv2
import math
import pydicom
import numpy as np
from glob import glob
from os.path import join
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import Sequence


class DataLoader_v2(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 n_patches=32,
                 patch_size=256,
                 augmentation=None,
                 **kwargs):
        super(DataLoader_v2, self).__init__()
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
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        indexes = self.indexes[idx: idx+1]
        if self.mode == "train":
            batch_x, batch_y = self.train_process(indexes)
        else:
            batch_x, batch_y = self.valid_process(indexes)
        return batch_x, batch_y

    def train_process(self, indexes):
        inputs = pydicom.dcmread(self.input_paths[indexes[0]]).pixel_array
        inputs = normalize(inputs).astype(np.float32)
        targets = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in self.target_paths[indexes[0]]]
        targets = normalize(np.stack(targets, axis=-1)).astype(np.float32)

        common_size = [self.n_patches, self.patch_size, self.patch_size]
        batch_x = np.zeros(common_size, dtype=np.float32)
        batch_y = np.zeros(common_size + [8], dtype=np.float32)

        bk = 0
        while bk < (self.n_patches // 2):
            patch_x, patch_y = random_crop(inputs, targets, self.patch_size)
            if not np.all(patch_y == 0):
                if self.augmentation:
                    aug = self.augmentation(image=patch_x, mask=patch_y)
                    batch_x[bk] = aug["image"]
                    batch_y[bk] = aug["mask"]
                else:
                    batch_x[bk] = patch_x
                    batch_y[bk] = patch_y
                bk += 1

        for i in range(bk, self.n_patches):
            patch_x, patch_y = random_crop(inputs, targets, self.patch_size)
            batch_x[i] = patch_x
            batch_y[i] = patch_y

        return batch_x[..., np.newaxis], batch_y

    def valid_process(self, indexes):
        inputs = pydicom.dcmread(self.input_paths[indexes[0]]).pixel_array
        inputs = normalize(inputs).astype(np.float32)
        targets = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in self.target_paths[indexes[0]]]
        targets = normalize(np.stack(targets, axis=-1)).astype(np.float32)

        common_size = [self.n_patches, self.patch_size, self.patch_size]
        batch_x = np.zeros(common_size, dtype=np.float32)
        batch_y = np.zeros(common_size + [8], dtype=np.float32)

        bk = 0
        while bk < (self.n_patches // 2):
            patch_x, patch_y = random_crop(inputs, targets, self.patch_size)
            if not np.all(patch_y == 0):
                batch_x[bk] = patch_x
                batch_y[bk] = patch_y
                bk += 1

        for i in range(bk, self.n_patches):
            patch_x, patch_y = random_crop(inputs, targets, self.patch_size)
            batch_x[i] = patch_x
            batch_y[i] = patch_y

        return batch_x[..., np.newaxis], batch_y


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def random_crop(x, y, crop_size, **kwargs):
    h, w = x.shape
    i = np.random.randint(0, (h-crop_size))
    j = np.random.randint(0, (w-crop_size))
    return x[i:(i+crop_size), j:(j+crop_size)], y[i:(i+crop_size), j:(j+crop_size), :]
