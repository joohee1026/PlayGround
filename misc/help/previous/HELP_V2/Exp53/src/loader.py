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

        valid_ratio = .2
        valid_length = int(len(sub_no) * valid_ratio)
        if mode == "train":
            self.sub_no = sub_no[valid_length:]
        else:
            self.sub_no = sub_no[:valid_length]

        self.mode = mode
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.reg_dir = reg_dir
        self.batch_size = batch_size

        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            self.input_size = input_size
        self.ch3 = ch3
        self.on_epoch_end()
        self.get_augment()

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.sub_no)

    def __len__(self):
        return math.ceil(len(self.sub_no) / self.batch_size)

    def get_augment(self):
        self.AUG = albu.Compose([
            albu.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
            albu.RandomGamma(gamma_limit=(100,400)),
            albu.GaussNoise(var_limit=(.001, .002)),
            albu.GaussianBlur(),
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(rotate_limit=5, shift_limit=.03, p=.8)
            # albu.VerticalFlip(),
            # albu.ElasticTransform(alpha=1, sigma=2, approximate=True),
            # albu.GridDistortion(num_steps=1)
        ], p=.5)

    def __getitem__(self, idx):
        sub_no = self.sub_no[idx*self.batch_size: (idx+1)*self.batch_size]

        bx, by = [], []
        for i in sub_no:
            image_path = join(self.data_dir, i, i+".dcm")
            image = pydicom.dcmread(image_path).pixel_array
            if self.ch3:
                image = get_3ch(image)
            else:
                image = histeq(image)
            image = normalize(image).astype(np.float32)

            seg_path = join(self.seg_dir, i+".npy")
            seg_mask = np.load(seg_path)/1.

            reg_path = join(self.reg_dir, i+".npy")
            reg_mask = np.load(reg_path)/1.

            image = np.concatenate([image, seg_mask, reg_mask], axis=-1).astype(np.float32)

            mask_paths = sorted(glob(join(self.data_dir, i, "*.png")))
            mask_lst = [np.array(cv2.imread(m, cv2.IMREAD_GRAYSCALE)) for m in mask_paths]
            mask = np.stack(mask_lst, axis=-1)
            mask[mask == 255] = 1.
            mask = mask.astype(np.float32)

            y = np.zeros(mask.shape[:2])
            for i in range(8):
                mask[..., i][mask[..., i] > 0] = 1.
                mask[..., i] *= (i + 1)
                y += mask[..., i]
                y[y >= (i + 1)] = i + 1
            y = (np.arange(y.max()) == y[..., None] - 1).astype(np.float32)
            bg = 1 - y.sum(axis=-1, keepdims=True)
            y = np.concatenate((y, bg), axis=-1)

            if self.mode == "train":
                aug = self.AUG(image=image, mask=y)
                image, y = aug["image"], aug["mask"]

            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
            y = cv2.resize(y, (self.input_size[1], self.input_size[0]), cv2.INTER_NEAREST)
            y[y >=.5] = 1.
            y[y <.5] = 0.

            bx.append(image)
            by.append(y)

        return np.array(bx), np.array(by).astype(np.float32)


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
