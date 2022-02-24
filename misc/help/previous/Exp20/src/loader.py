import cv2
import math
import pydicom
import numpy as np
from glob import glob
from os.path import join

from tensorflow.keras.utils import Sequence


class PatchDataLoader(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 n_patches=8,
                 img_size=512,
                 augmentation=None,
                 **kwargs):
        super(PatchDataLoader, self).__init__()
        assert mode in ["train", "valid"]

        data_dirs = glob(join(data_dir, "*"))
        input_paths = [glob(join(f, "*.dcm"))[0] for f in data_dirs]
        target_paths = [sorted(glob(join(f, "*.png"))) for f in data_dirs]

        valid_ratio = .1
        valid_length = int(len(input_paths) * valid_ratio)
        train_X_paths, valid_X_paths = input_paths[valid_length:], input_paths[:valid_length]
        train_y_paths, valid_y_paths = target_paths[valid_length:], target_paths[:valid_length]

        self.n_patches = n_patches
        self.img_size = img_size
        self.augmentation = augmentation

        self.mode = mode
        if mode == "train":
            self.input_paths = train_X_paths
            self.target_paths = train_y_paths
            self.indexes = np.arange(len(self.input_paths))
            self.batch_size = 1
        else:
            self.img_devel, self.mask_devel = self.get_development(valid_X_paths, valid_y_paths)
            self.indexes = np.arange(len(self.img_devel))
            self.batch_size = n_patches

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes) // self.batch_size


    def __getitem__(self, idx):
        if self.mode == "train":
            indexes = self.indexes[idx: (idx + 1)]
            bx, by = self.patch_prep(self.input_paths[indexes[0]], self.target_paths[indexes[0]])
            return bx, by
        else:
            indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]
            bx, by = [], []
            for i in indexes:
                bx.append(self.img_devel[i])
                by.append(self.mask_devel[i])
            return np.array(bx), np.array(by)


    def patch_prep(self, img_path, mask_path):
        img = pydicom.dcmread(img_path).pixel_array
        img = normalize(img).astype(np.float32)

        mask_lst = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in mask_path]
        mask = np.zeros(img.shape)
        for i in range(8):
            mask[..., i][mask[..., i] > 0] = 1.
            mask[..., i] *= (i + 1)
            y += mask[..., i]
            y[y >= (i + 1)] = i + 1
        y = (np.arange(y.max()) == y[..., None] - 1).astype(np.float32)
        bg = 1 - y.sum(axis=-1, keepdims=True)
        mask = np.concatenate((y, bg), axis=-1)

        bx, by = [], []
        roi_img, roi_mask = roi_crop(img, mask)
        for i in range(self.n_patches//2):
            px, py = self.random_roi_crop(roi_img, roi_mask)
            if self.augmentation:
                aug = self.augmentation(image=px, mask=py)
                px, py = aug["image"], aug["mask"]
            bx.append(px)
            by.append(py)

        for i in range(self.n_patches//2):
            px, py = self.random_roi_crop(img, mask)
            if self.augmentation:
                aug = self.augmentation(image=px, mask=py)
                px, py = aug["image"], aug["mask"]
            bx.append(px)
            by.append(py)

        return np.array(bx)[..., np.newaxis], np.array(by)

    def random_roi_crop(self, x, y):
        h, w = x.shape
        i = np.random.randint(0, (h - self.img_size))
        j = np.random.randint(0, (w - self.img_size))
        return x[i:(i + self.img_size), j:(j + self.img_size)], y[i:(i + self.img_size), j:(j + self.img_size), :]

    def get_development(self, img_paths, mask_paths):
        img_patches, mask_patches = [], []
        for img_path, mask_path in zip(img_paths, mask_paths):
            img = pydicom.dcmread(img_path).pixel_array
            img = random_histeq(img, 5)
            img = normalize(img).astype(np.float32)

            mask_lst = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in mask_path]
            mask = np.zeros(img.shape)
            for i, m in enumerate(mask_lst):
                m[m>0] = 1.
                m *= (i+1)
                mask += m
                mask[mask >= (i+1)] = i+1
            mask = (np.arange(mask.max()) == mask[..., None] - 1).astype(np.float32)
            bg = 1 - mask.sum(axis=-1, keepdims=True)

            img = img[..., np.newaxis]
            mask = np.concatenate((mask, bg), axis=-1)

            stride = self.img_size - 2*(self.img_size//9)
            for i in range(100, img.shape[0]-self.img_size, stride):
                for j in range(100, img.shape[1]-self.img_size, stride):
                    img_patches.append(img[
                                       i:i+self.img_size,
                                       j:j+self.img_size,
                                       :])
                    mask_patches.append(mask[
                                       i:i+self.img_size,
                                       j:j+self.img_size,
                                       :])

        return img_patches, mask_patches


def roi_crop(x, y):
    y_ = y[...,:8].sum(axis=-1, keepdims=True)
    y_[y_ != 0] = 1.

    nonzero = np.argwhere(y_)
    top_edge, bottom_edge = nonzero.min(axis=0), nonzero.max(axis=0)

    x = x[
        top_edge[0]-200: bottom_edge[0]+200,
        top_edge[1]-200: bottom_edge[1]+50
    ]
    y = y[
        top_edge[0]-200: bottom_edge[0]+200,
        top_edge[1]-200: bottom_edge[1]+50,
        :
    ]
    return x, y


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
