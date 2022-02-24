import cv2
import math
import pydicom
import numpy as np
import albumentations as albu

from glob import glob
from os.path import join

from tensorflow.keras.utils import Sequence


class SDataLoader(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 batch_size,
                 input_size,
                 roi=None
                 ):
        super(SDataLoader, self).__init__()
        assert mode in ["train", "valid"]

        data_dirs = glob(join(data_dir, "*"))
        image_paths = [glob(join(f, "*.dcm"))[0] for f in data_dirs]
        mask_paths = [sorted(glob(join(f, "*.png"))) for f in data_dirs]

        self.mode = mode

        valid_ratio = .2
        valid_length = int(len(image_paths) * valid_ratio)
        if mode == "train":
            self.image_paths = image_paths[valid_length:]
            self.mask_paths = mask_paths[valid_length:]
        else:
            self.image_paths = image_paths[:valid_length]
            self.mask_paths = mask_paths[:valid_length]

        self.indexes = np.arange(len(self.image_paths))
        self.batch_size = batch_size
        self.input_size = input_size
        # self.input_size_lst = [
        #     (input_size, input_size),
        #     (input_size-96, input_size+96) if input_size == 512 else (input_size-128, input_size+128)
        # ]

        self.roi = roi
        self.on_epoch_end()
        self.get_augment()

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]

        # if self.mode == "train":
        #     size_idx = np.random.choice([0, 1])
        #     if size_idx == 0:
        #         self.input_size = self.input_size_lst[0]
        #     else:
        #         self.input_size = self.input_size_lst[1]

        bx, by = [], []
        for i in indexes:
            image, mask = prep(self.image_paths[i], self.mask_paths[i])

            if self.roi:
                image, mask = crop(image, mask)

            if self.mode == "train":
                image, mask = rotate_90(image, mask)
                aug = self.AUG(image=image, mask=mask)
                image, mask = aug["image"], aug["mask"]

                # image = cv2.resize(image, (self.input_size[0], self.input_size[1]))
                # mask = cv2.resize(mask, (self.input_size[0], self.input_size[1]), cv2.INTER_NEAREST)
                image = cv2.resize(image, (self.input_size, self.input_size))
                mask = cv2.resize(mask, (self.input_size, self.input_size), cv2.INTER_NEAREST)
            else:
                image = cv2.resize(image, (self.input_size, self.input_size))
                mask = cv2.resize(mask, (self.input_size, self.input_size), cv2.INTER_NEAREST)

            mask[mask >=.5] = 1.
            mask[mask <.5] = 0.
            bx.append(image)
            by.append(mask)
        return np.array(bx)[...,np.newaxis], np.array(by).astype(np.float32)

    def get_augment(self):
        self.AUG = albu.Compose([
            albu.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
            albu.RandomGamma(gamma_limit=(100,400)),
            # albu.GaussNoise(var_limit=(.001, .002)),
            # albu.GaussianBlur(),
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            # albu.ElasticTransform(alpha=1, sigma=2, approximate=True, p=.3),
            # albu.GridDistortion(num_steps=1, p=.3)
        ], p=.8)


def prep(xp, yp):
    x = histeq(pydicom.dcmread(xp).pixel_array)
    x = normalize(x).astype(np.float32)

    y_lst = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in yp]
    y = np.array(y_lst)
    y[y == 255] = 1.
    y = np.swapaxes(y, 0, 2)
    y = np.swapaxes(y, 0, 1)
    return x, y


def crop(x, y, rand=True):
    (e1, e2), (e3, e4) = get_roi(y)
    if rand:
        c = [i for i in range(16, 100)]
        v1, v3 = np.random.choice(c,1)[0], np.random.choice(c,1)[0]
        v2, v4 = np.random.choice(c, 1)[0], np.random.choice(c, 1)[0]
        return x[e1-v1:e3+v3, e2-v2:e4+v4], y[e1-v1:e3+v3, e2-v2:e4+v4, :]
    else:
        return x[e1-16:e3+16, e2-16:e4+16], y[e1-16:e3+16, e2-16:e4+16, :]


def get_roi(y):
    # y : 8 channels
    bg_ = np.sum(y, axis=-1)
    bg_[bg_ == 0] = 255.
    bg_[bg_ < 255.] = 0.
    bg_[bg_ == 255] = 1.
    bg = 1 - bg_
    bg = bg.astype(np.float32)

    nonzero = np.argwhere(bg)
    (e1, e2), (e3, e4) = nonzero.min(axis=0), nonzero.max(axis=0)
    return (e1, e2), (e3, e4)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def histeq(x, rand=None):
    if rand is None:
        return cv2.createCLAHE(tileGridSize=(1,1)).apply(x)
    else:
        g = np.random.randint(1, 5)
        return cv2.createCLAHE(tileGridSize=(g,g)).apply(x)


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
