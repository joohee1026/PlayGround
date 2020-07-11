import cv2
import math
import pydicom
import numpy as np
import albumentations as albu

from glob import glob
from os.path import join

from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 batch_size=8,
                 input_size=512,
                 roi=None,
                 **kwargs):
        super(DataLoader, self).__init__()
        assert mode in ["train", "valid"]

        data_dirs = glob(join(data_dir, "*"))
        image_paths = [glob(join(f, "*.dcm"))[0] for f in data_dirs]
        mask_paths = [sorted(glob(join(f, "*.png"))) for f in data_dirs]

        valid_ratio = .2
        valid_length = int(len(image_paths) * valid_ratio)
        self.mode = mode
        if mode == "train":
            self.image_paths = image_paths[valid_length:]
            self.mask_paths = mask_paths[valid_length:]
        else:
            self.image_paths = image_paths[:valid_length]
            self.mask_paths = mask_paths[:valid_length]

        self.indexes = np.arange(len(self.image_paths))
        self.batch_size = batch_size
        self.input_size = input_size
        self.on_epoch_end()
        self.get_augment()

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]

        bx, by = [], []
        for i in indexes:
            image = self.x_lr_prep(self.image_paths[i])
            mask = self.y_lr_prep(self.mask_paths[i])
            if self.mode == "train":
                image, mask = rotate_90(image, mask)
                aug = self.AUG(image=image, mask=mask)
                bx.append(aug["image"][..., np.newaxis])
                by.append(aug["mask"])
            else:
                bx.append(image[..., np.newaxis])
                by.append(mask)
        return np.array(bx), np.array(by)

    def x_lr_prep(self, path):
        img = random_histeq(pydicom.dcmread(path).pixel_array, 5)
        return normalize(cv2.resize(img, (self.input_size, self.input_size))).astype(np.float32)

    def y_lr_prep(self, path):
        img_lst = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in path]
        img = np.zeros(img_lst[0].shape)
        for i, m in enumerate(img_lst):
            m[m>0] = 1.
            m *= (i+1)
            img += m
            img[img >= (i+1)] = i+1
        img = (np.arange(img.max()) == img[..., None]-1).astype(np.float32)
        bg = 1 - img.sum(axis=-1, keepdims=True)
        img = np.concatenate((img, bg), axis=-1)

        img = cv2.resize(img, (self.input_size, self.input_size), cv2.INTER_NEAREST)
        img[img>=.5] = 1.
        img[img<.5] = 0.

        return img

    def get_augment(self):
        self.AUG = albu.Compose([
            albu.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
            albu.RandomGamma(gamma_limit=(100,400)),
            albu.GaussNoise(var_limit=(.001, .002)),
            albu.GaussianBlur(),
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ElasticTransform(alpha=1, sigma=2, approximate=True, p=.1),
            albu.GridDistortion(num_steps=1, p=.1)
        ], p=.8)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def random_histeq(x, max_n):
    if np.random.random() < .5:
        g = np.random.randint(1, max_n)
        return cv2.createCLAHE(tileGridSize=(g, g)).apply(x)
    else:
        return x


def rotate_90(img, mask):
    """
    :param img: (h, w) dim
    :param mask: (h, w, c) dim
    """
    if np.random.random() < .5:
        img = img.T
        mask = np.transpose(mask, (1,0,2))
        return img, mask
    else:
        return img, mask