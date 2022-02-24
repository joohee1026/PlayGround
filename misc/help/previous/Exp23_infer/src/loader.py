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
        auxy1, auxy2, auxy3 = [], [], []
        for i in indexes:
            image, mask = self.prep(self.image_paths[i], self.mask_paths[i])
            if self.mode == "train":
                image, mask = rotate_90(image, mask)
                aug = self.AUG(image=image, mask=mask)
                image, mask = aug["image"], aug["mask"]

            bx.append(cv2.resize(image, (self.input_size, self.input_size)))
            y = cv2.resize(mask, (self.input_size, self.input_size))
            y[y>0] = 1.
            by.append(y)
            aux3 = cv2.resize(mask, (self.input_size//2, self.input_size//2))
            aux3[aux3>0] = 1.
            auxy3.append(aux3)
            aux2 = cv2.resize(mask, (self.input_size//4, self.input_size//4))
            aux2[aux2>0] = 1.
            auxy2.append(aux2)
            aux1 = cv2.resize(mask, (self.input_size//8, self.input_size//8))
            aux1[aux1>0] = 1.
            auxy1.append(aux1)

        return np.array(bx)[..., np.newaxis], [np.array(auxy1), np.array(auxy2), np.array(auxy3), np.array(by)]

    def prep(self, xp, yp):
        x = random_histeq(pydicom.dcmread(xp).pixel_array, 5)
        x = normalize(x).astype(np.float32)

        y_lst = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in yp]
        y = np.zeros(y_lst[0].shape)
        for i, m in enumerate(y_lst):
            m[m>0] = 1.
            m *= (i+1)
            y += m
            y[y>=(i+1)] = i+1
        y = (np.arange(y.max()) == y[..., None]-1).astype(np.float32)
        bg = 1 - y.sum(axis=-1, keepdims=True)
        y = np.concatenate((y, bg), axis=-1)
        return x, y

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