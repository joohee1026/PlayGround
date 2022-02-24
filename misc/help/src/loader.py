import cv2
import math
import pydicom
import numpy as np
from glob import glob
from os.path import join
from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    def __init__(self,
                 data_dir,
                 batch_size=1,
                 n_patches=32,
                 patch_size=512,
                 augment_prob=.0,
                 **kwargs):
        super(DataLoader, self).__init__()
        data_dirs = glob(join(data_dir, "*"))
        self.input_paths = [glob(join(f, "*.dcm"))[0] for f in data_dirs]
        self.target_paths = [sorted(glob(join(f, "*.png"))) for f in data_dirs]

        # print("inputs load ..")
        # self.inputs = [mm_norm(pydicom.dcmread(f).pixel_array).astype(np.float32) for f in input_paths]
        # print("input loaded")
        #
        # print("targets load .. ")
        # self.targets = []
        # for t in target_paths:
        #     target = np.stack([np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in t], axis=-1)
        #     self.targets.append(mm_norm(target).astype(np.float32))
        # print("target loaded")

        self.indexes = np.arange(len(self.input_paths))
        self.batch_size = batch_size
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.augment_prob = augment_prob

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size : (idx+1)*self.batch_size]
        inputs = mm_norm(pydicom.dcmread(self.input_paths[indexes[0]]).pixel_array).astype(np.float32)
        targets = np.stack([np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in self.target_paths[indexes[0]]], axis=-1)
        targets = mm_norm(targets)

        # inputs = [self.inputs[i] for i in indexes]
        # targets = [self.targets[i] for i in indexes]

        common_size = [self.batch_size*self.n_patches, self.patch_size, self.patch_size]
        batch_x = np.zeros(common_size + [1], dtype=np.float32)
        batch_y = np.zeros(common_size + [8], dtype=np.float32)

        for i in range(self.n_patches):
            patch_x, patch_y = random_crop(inputs, targets, self.patch_size)
            batch_x[i] = patch_x[..., np.newaxis]
            batch_y[i] = patch_y

        return batch_x, batch_y


def mm_norm(x):
    return (x - x.min()) / (x.max() - x.min())


def random_crop(x, y, crop_size, **kwargs):
    h, w = x.shape
    i = np.random.randint(0, (h-crop_size))
    j = np.random.randint(0, (w-crop_size))
    return x[i:(i+crop_size), j:(j+crop_size)], y[i:(i+crop_size), j:(j+crop_size)]
