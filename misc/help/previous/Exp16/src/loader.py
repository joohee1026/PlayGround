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
                 use_patch=False,
                 batch_size=4,
                 img_size=512,
                 augmentation=None,
                 **kwargs):
        super(DataLoader, self).__init__()
        assert mode in ["train", "valid"]

        data_dirs = glob(join(data_dir, "*"))
        input_paths = [glob(join(f, "*.dcm"))[0] for f in data_dirs]
        target_paths = [sorted(glob(join(f, "*.png"))) for f in data_dirs]

        valid_ratio = .1
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
        self.use_patch = use_patch

        if self.use_patch:
            self.batch_size = 1
            self.n_patches = batch_size
        else:
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

        if self.use_patch:
            bx, by = self.patch_prep(self.input_paths[indexes[0]], self.target_paths[indexes[0]])
            return bx, by
        else:
            bx, by = [], []
            for i in indexes:
                image = self.x_lr_prep(self.input_paths[i])
                mask = self.y_lr_prep(self.target_paths[i])
                if self.augmentation:
                    image, mask = rotate_90(image, mask)
                    aug = self.augmentation(image=image, mask=mask)
                    image = normalize(aug["image"])[..., np.newaxis]
                    mask = normalize(aug["mask"])
                    bx.append(image)
                    by.append(mask)
                else:
                    bx.append(image[..., np.newaxis])
                    by.append(mask)
            return np.array(bx), np.array(by)

    def x_lr_prep(self, path):
        img = pydicom.dcmread(path).pixel_array
        # img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        img = random_histeq(img, 7)
        return normalize(cv2.resize(img, (self.img_size, self.img_size))).astype(np.float32)

    def y_lr_prep(self, path):
        img = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in path]
        img = normalize(np.stack(img, axis=-1)).astype(np.float32)
        bg = 1 - img.sum(axis=-1, keepdims=True)
        bg = clip_0_1(bg)
        img = np.concatenate((img, bg), axis=-1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img[img>0] = 1.
        return img

    def patch_prep(self, input_path, target_path):
        img_x = pydicom.dcmread(input_path).pixel_array
        img_x = normalize(img_x).astype(np.float32)[..., np.newaxis]
        img_y = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in target_path]
        img_y = normalize(np.stack(img_y, axis=-1)).astype(np.float32)
        bg_y = 1 - img_y.sum(axis=-1, keepdims=True)
        bg_y = clip_0_1(bg_y)
        img_y = np.concatenate((img_y, bg_y), axis=-1)

        bx, by, inr = [], [], 0
        while inr < (self.n_patches // 2):
            px, py = self.random_crop(img_x, img_y)
            if not np.all(py[...,:8] == 0):
                if self.augmentation:
                    aug = self.augmentation(image=px, mask=py)
                    image = normalize(aug["image"])
                    mask = normalize(aug["mask"])
                    bx.append(image)
                    by.append(mask)
                else:
                    bx.append(px)
                    by.append(py)
                inr += 1

        for i in range(inr, self.n_patches):
            px, py = self.random_crop(img_x, img_y)
            bx.append(px)
            by.append(py)

        return np.array(bx), np.array(by)

    def random_crop(self, x, y):
        h, w, _ = x.shape
        i = np.random.randint(0, (h - self.img_size))
        j = np.random.randint(0, (w - self.img_size))
        return x[i:(i + self.img_size), j:(j + self.img_size)], y[i:(i + self.img_size), j:(j + self.img_size), :]


class DataLoader_RoI(Sequence):
    def __init__(self,
                 mode,
                 data_dir="/data/volume/Datasets/ROI200/TRAIN",
                 max_pad=200,
                 batch_size=8,
                 img_size=512,
                 augmentation=None):
        super(DataLoader_RoI, self).__init__()
        assert mode in ["train", "valid"]

        input_paths = sorted(glob(join(data_dir, "*")))
        valid_ratio = .2
        valid_length = int(len(input_paths) * valid_ratio)
        train_paths, valid_paths = input_paths[valid_length:], input_paths[:valid_length]

        self.mode = mode
        if mode == "train":
            self.input_paths = train_paths
        else:
            self.input_paths = valid_paths

        self.indexes = np.arange(len(train_paths))
        self.max_pad = max_pad
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
            npz = np.load(self.input_paths[i])
            image, mask = npz["image"], npz["mask"]
            mask[mask>0] = 1.
            # mask[mask==0] = 0.1
            # mask[mask==1] = 0.9

            if self.mode == "train":
                image, mask = self.random_pad_resize(image, mask)
                image, mask = rotate_90(image, mask)
                image = random_histeq(image, 5)
                aug = self.augmentation(image=image, mask=mask)
                image = normalize(aug["image"])[..., np.newaxis]
                bx.append(image.astype(np.float32))
                by.append(mask)
            else:
                image, mask = self.pad_resize(image, mask)
                bx.append(normalize(image[..., np.newaxis]).astype(np.float32))
                by.append(mask)
        return np.array(bx), np.array(by)

    def random_pad_resize(self, image, mask):
        orig_shape = image.shape
        v1, v2, v3, v4 = np.random.choice(self.max_pad-16, 4)
        image = image[v1:orig_shape[0]-v2, v3:orig_shape[1]-v4]
        mask = mask[v1:orig_shape[0]-v2, v3:orig_shape[1]-v4, :]
        return cv2.resize(image, (self.img_size, self.img_size)), cv2.resize(mask, (self.img_size, self.img_size))

    def pad_resize(self, image, mask, del_pad=100):
        orig_shape = image.shape
        image = image[del_pad:orig_shape[0]-del_pad, del_pad:orig_shape[1]-del_pad]
        mask = mask[del_pad:orig_shape[0]-del_pad, del_pad:orig_shape[1]-del_pad]
        return cv2.resize(image, (self.img_szie, self.img_size)), cv2.resize(mask, (self.img_size, self.img_size))


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def clip_0_1(x):
    x[x < 0] = 0
    return x


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

