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
                 aux=True,
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
        self.roi = roi
        self.aux = aux
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


class PatchDataLoader(Sequence): # test 필요
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
        if self.mode == "train":
            return len(self.indexes) // self.batch_size
        else:
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
        mask = np.concatenate((mask, bg), axis=-1)

        bx, by = [], []
        roi_img, roi_mask = roi_crop(img, mask)
        for i in range(self.n_patches//2):
            px, py = self.random_roi_crop(roi_img, roi_mask)
            if self.augmentation:
                px, py = rotate_90(px, py)
                aug = self.augmentation(image=px, mask=py)
                px, py = aug["image"], aug["mask"]
            bx.append(px)
            by.append(py)

        for i in range(self.n_patches//2):
            px, py = self.random_roi_crop(img, mask)
            if self.augmentation:
                px, py = rotate_90(px, py)
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


class PatchDataLoader_randdev(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 n_patches=8,
                 img_size=512,
                 augmentation=None,
                 **kwargs):
        super(PatchDataLoader_randdev, self).__init__()
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
        else:
            self.input_paths = valid_X_paths
            self.target_paths = valid_y_paths
        self.indexes = np.arange(len(self.input_paths))
        self.batch_size = 1
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx: (idx + 1)]
        bx, by = self.patch_prep(self.input_paths[indexes[0]], self.target_paths[indexes[0]])
        return bx, by

    def patch_prep(self, img_path, mask_path):
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
        mask = np.concatenate((mask, bg), axis=-1)

        bx, by = [], []
        roi_img, roi_mask = roi_crop(img, mask)
        for i in range(self.n_patches//2):
            px, py = self.random_roi_crop(roi_img, roi_mask)
            if self.augmentation:
                px, py = rotate_90(px, py)
                aug = self.augmentation(image=px, mask=py)
                px, py = aug["image"], aug["mask"]
            bx.append(px)
            by.append(py)

        for i in range(self.n_patches//2):
            px, py = self.random_roi_crop(img, mask)
            if self.augmentation:
                px, py = rotate_90(px, py)
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

        bx, by = [], []
        for i in indexes:
            image, mask = prep(self.image_paths[i], self.mask_paths[i])

            if self.roi:
                image, mask = crop(image, mask)

            if self.mode == "train":
                image, mask = rotate_90(image, mask)
                aug = self.AUG(image=image, mask=mask)
                image, mask = aug["image"], aug["mask"]
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
            albu.GaussNoise(var_limit=(.001, .002)),
            albu.GaussianBlur(),
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ElasticTransform(alpha=1, sigma=2, approximate=True, p=.3),
            albu.GridDistortion(num_steps=1, p=.3)
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
    y_ = y[...,:8].sum(axis=-1)

    nonzero = np.argwhere(y_)
    top_edge, bottom_edge = nonzero.min(axis=0), nonzero.max(axis=0)
    return top_edge, bottom_edge


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def histeq(x, rand=None):
    if rand is None:
        return cv2.createCLAHE(tileGridSize=(3,3)).apply(x)
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
