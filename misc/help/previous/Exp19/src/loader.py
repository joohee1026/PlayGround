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
                 batch_size=4,
                 img_size=512,
                 augmentation=None,
                 **kwargs):
        super(DataLoader, self).__init__()
        assert mode in ["train", "valid"]

        data_dirs = glob(join(data_dir, "*"))
        input_paths = [glob(join(f, "*.dcm"))[0] for f in data_dirs]
        target_paths = [sorted(glob(join(f, "*.png"))) for f in data_dirs]

        valid_ratio = .2
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
        self.batch_size = batch_size

        self.img_size = img_size
        self.augmentation = augmentation
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]

        bx, by = [], []
        for i in indexes:
            img, mask = self.crop_prep(self.input_paths[i], self.target_paths[i])
            if self.augmentation:
                img, mask = rotate_90(img, mask)
                aug = self.augmentation(image=img, mask=mask)
                img, mask = aug["image"], aug["mask"]
                bx.append(img)
                by.append(mask)
            else:
                bx.append(img)
                by.append(mask)
        return np.array(bx)[...,np.newaxis], np.array(by)

    def crop_prep(self, img_path, mask_path):
        img = pydicom.dcmread(img_path).pixel_array
        img = random_histeq(img, 5)

        mask_lst = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in mask_path]
        mask = np.zeros(img.shape)
        for i, m in enumerate(mask_lst):
            m[m>0] = 1
            m *= (i+1)
            mask += m
            mask[mask >= (i+1)] = i+1
        mask = (np.arange(mask.max()) == mask[..., None] - 1).astype(np.float32)
        bg = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, bg), axis=-1)

        img, mask = random_crop(img, mask)
        img = normalize(cv2.resize(img, (self.img_size, self.img_size))).astype(np.float32)
        mask = cv2.resize(mask, (self.img_size, self.img_size), cv2.INTER_NEAREST)
        mask[mask <1] = 0.
        return img, mask


def random_crop(img, mask):
    # v1 = np.random.choice(400, 1)[0] + 300
    # v2 = np.random.choice(300, 1)[0] + 100
    # v3 = np.random.choice(300, 1)[0] + 200
    # v4 = np.random.choice(200, 1)[0] + 200
    v1 = np.random.choice(700, 1)[0]
    v2 = np.random.choice(400, 1)[0]
    v3 = np.random.choice(500, 1)[0]
    v4 = np.random.choice(400, 1)[0]
    return img[v1:-v2, v3:-v4], mask[v1:-v2, v3:-v4, :]


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
