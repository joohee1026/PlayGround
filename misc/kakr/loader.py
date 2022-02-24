import os
import cv2
import numpy as np
from operator import itemgetter
from scipy.ndimage import rotate

import tensorflow as tf
from tensorflow.keras.utils import Sequence

def random_flip(img):
    def vertical_flip(img):
        return img[::-1, :, :]
    def horizontal_flip(img):
        return img[:, ::-1, :]
    def double_flip(img):
        return vertical_flip(horizontal_flip(img))

    flip_lst = [vertical_flip, horizontal_flip, double_flip]
    flip_func = np.random.choice(flip_lst)
    return flip_func(img)

def random_rotate(img, angle=None, mode='nearest'):
    if angle is None:
        angle_range = (0, 360)
        angle = np.random.randint(*(angle_range))
    return rotate(img, angle, reshape=False, mode=mode)

class Dataloader(utils.Sequence):
    def __init__(self,
                 img_file_lst,
                 img_file_path,
                 img_labels,
                 batch_size,
                 n_classes,
                 img_size=300,
                 prob_flip=.5,
                 prob_rotate=.5,
                 prob_cutmix=.8,
                 shuffle=True):
        super(Dataloader, self).__init__()
        self.img_file_paths = [os.path.join(img_file_path, f) for f in img_file_lst]
        self.img_labels = img_labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.img_size = img_size
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.prob_cutmix = prob_cutmix
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.img_file_paths))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.img_file_paths) // self.batch_size

    def __getitem__(self, ind: int) -> np.array:
        has_cutmix = [True if np.random.random() < self.prob_cutmix else False for _ in range(self.batch_size)]

        indexes = self.indexes[ind * self.batch_size : (ind+1) * self.batch_size]
        batch_x = np.zeros((self.batch_size, self.img_size, self.img_size, 3))
        batch_y = np.zeros((self.batch_size, self.n_classes))

        for i, idx in enumerate(indexes):
            img = cv2.imread(self.img_file_paths[idx], cv2.IMREAD_COLOR)
            img = self._flip_rotate(self._resize(img))
            if has_cutmix[i] is False:
                batch_x[i] = img
                batch_y[i][self.img_labels[idx]-1] = 1.
            else:
                mix_i = np.random.randint(self.n_classes, size=1)[0]
                mix_img = cv2.imread(self.img_file_paths[mix_i], cv2.IMREAD_COLOR)
                mix_img = self._flip_rotate(self._resize(mix_img))
                batch_x[i] = self._cutmix(img, mix_img)

                if idx == mix_i:
                    batch_y[i][self.img_labels[idx]-1] = 1.
                else:
                    batch_y[i][self.img_labels[idx]-1] = .5
                    batch_y[i][self.img_labels[mix_i]-1] = .5

        batch_x = self._min_max_norm(batch_x).astype(np.float32)
        return batch_x, batch_y

    def _resize(self, img):
        return cv2.resize(img, (self.img_size, self.img_size))

    def _flip_rotate(self, img):
        if np.random.random() < self.prob_flip:
            img = random_flip(img)
        if np.random.random() < self.prob_rotate:
            img = random_rotate(img)
        return img

    def _cutmix(self, img1, img2):
        img1 = img1[:, :self.img_size//2, :]
        img2 = img2[:, self.img_size//2:, :]
        img = np.concatenate((img1, img2), axis=1)
        return img

    def _min_max_norm(self, img):
        return (img - img.min()) / (img.max() - img.min())
