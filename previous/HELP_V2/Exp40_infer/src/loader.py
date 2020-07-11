import cv2
import math
import pydicom
import numpy as np
import albumentations as albu

from glob import glob
from os import listdir
from os.path import join

from tensorflow.keras.utils import Sequence


class S3DataLoader(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 seg_dir,
                 reg_dir,
                 batch_size,
                 input_size,
                 ch3=True
                 ):
        super(S3DataLoader, self).__init__()
        assert mode in ["train", "valid"]

        sub_no = sorted(listdir(data_dir))

        image_paths = [glob(join(data_dir, s, "*.dcm"))[0] for s in sub_no]
        seg_paths = [glob(join(seg_dir, s, "*.npy"))[0] for s in sub_no]
        reg_paths = [glob(join(reg_dir, s, "*.npy"))[0] for s in sub_no]
        mask_paths = [sorted(glob(join(data_dir, s, "*.png"))) for s in sub_no]

        valid_ratio = .2
        valid_length = int(len(image_paths) * valid_ratio)
        if mode == "train":
            self.image_paths = image_paths[valid_length:]
            self.seg_paths = seg_paths[valid_length:]
            self.reg_paths = reg_paths[valid_length:]
            self.mask_paths = mask_paths[valid_length:]
        else:
            self.image_paths = image_paths[:valid_length]
            self.seg_paths = seg_paths[:valid_length]
            self.reg_paths = reg_paths[:valid_length]
            self.mask_paths = mask_paths[:valid_length]

        self.mode = mode
        self.indexes = np.arange(len(self.image_paths))
        self.batch_size = batch_size
        self.input_size = input_size
        self.ch3 = ch3
        self.on_epoch_end()
        self.get_augment()

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def get_augment(self):
        self.AUG = albu.Compose([
            albu.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
            albu.RandomGamma(gamma_limit=(100,400)),
            albu.GaussNoise(var_limit=(.001, .002)),
            albu.GaussianBlur(),
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ElasticTransform(alpha=1, sigma=2, approximate=True, p=.2),
            albu.GridDistortion(num_steps=1, p=.2)
        ], p=.5)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]

        bx, by = [], []
        for i in indexes:
            image = pydicom.dcmread(self.image_path[i]).pixel_array
            if self.ch3:
                image = get_3ch(image)
            else:
                image = histeq(image)
            image = normalize(image).astype(np.float32)

            seg_mask = np.load(self.seg_paths[i])/255.
            reg_mask = np.load(self.reg_paths[i])/255.
            image = np.concatenate([image, seg_mask, reg_mask], axis=-1)

            mask_lst = [np.array(cv2.imread(m, cv2.IMREAD_GRAYSCALE)) for m in self.mask_paths[i]]
            mask = np.stack(mask_lst, axis=-1)
            mask[mask == 255] = 1.

            if self.mode == "train":
                aug = self.AUG(image=image, mask=mask)
                image, mask = aug["image"], aug["mask"]

            image = cv2.resize(image, (self.input_size, self.input_size))
            mask = cv2.resize(mask, (self.input_size, self.input_size), cv2.INTER_NEAREST)
            mask[mask >=.5] = 1.
            mask[mask <.5] = 0.

            bx.append(image)
            by.append(mask)

        return np.array(bx), np.array(by).astype(np.float32)


class SDataLoader(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 batch_size,
                 input_size,
                 ch3=True,
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
        self.ch3 = ch3
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
            image, mask = prep(self.image_paths[i], self.mask_paths[i], self.ch3)

            if self.mode == "train":
                # image, mask = rotate_90(image, mask)
                aug = self.AUG(image=image, mask=mask)
                image, mask = aug["image"], aug["mask"]

            image = cv2.resize(image, (self.input_size, self.input_size))
            mask = cv2.resize(mask, (self.input_size, self.input_size), cv2.INTER_NEAREST)
            mask[mask >=.5] = 1.
            mask[mask <.5] = 0.

            if not self.ch3:
                image = image[..., np.newaxis]
            bx.append(image)
            by.append(mask)
        return np.array(bx), np.array(by).astype(np.float32)

    def get_augment(self):
        self.AUG = albu.Compose([
            albu.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
            albu.RandomGamma(gamma_limit=(100,400)),
            albu.GaussNoise(var_limit=(.001, .002)),
            albu.GaussianBlur(),
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ElasticTransform(alpha=1, sigma=2, approximate=True, p=.3),
            albu.GridDistortion(num_steps=1, p=.3)
        ], p=.5)


def prep(xp, yp, ch3=False):
    if ch3:
        x = pydicom.dcmread(xp).pixel_array
        x = get_3ch(x)
    else:
        x = histeq(pydicom.dcmread(xp).pixel_array)

    x = normalize(x).astype(np.float32)

    y_lst = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in yp]
    y = np.array(y_lst)
    y[y == 255] = 1.
    y = np.swapaxes(y, 0, 2)
    y = np.swapaxes(y, 0, 1)
    return x, y


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def histeq(x, rand=None):
    if rand is None:
        return cv2.createCLAHE(tileGridSize=(3,3)).apply(x)
    elif isinstance(rand, int):
        return cv2.createCLAHE(tileGridSize=(rand, rand)).apply(x)
    else:
        g = np.random.randint(1, 5)
        return cv2.createCLAHE(tileGridSize=(g,g)).apply(x)


def get_3ch(x, c1=3, c2=7):
    x0 = x
    x1 = histeq(x, c1)
    x2 = histeq(x, c2)
    return np.stack([x0, x1, x2], axis=-1)


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
