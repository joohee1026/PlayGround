import cv2
import math
import pydicom
import numpy as np
import albumentations as albu
from glob import glob
from os.path import join

from tensorflow.keras.utils import Sequence


class CrdSegSDataLoader(Sequence):
    def __init__(self,
                 mode,
                 data_dir,
                 batch_size,
                 input_size,
                 roi=None,
                 only_crd=None,
                 ):
        super(CrdSegSDataLoader, self).__init__()
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
        self.only_crd = only_crd
        self.on_epoch_end()
        self.get_augment()

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]

        bx, by, bc = [], [], []
        for i in indexes:
            image, mask = prep(self.image_paths[i], self.mask_paths[i])

            if self.roi:
                image, mask = crop(image, mask)

            if self.mode == "train":
                # image, mask = rotate_90(image, mask)
                aug = self.AUG(image=image, mask=mask)
                image, mask = aug["image"], aug["mask"]
            image = cv2.resize(image, (self.input_size, self.input_size))
            mask = cv2.resize(mask, (self.input_size, self.input_size), cv2.INTER_NEAREST)
            mask[mask >=.5] = 1.
            mask[mask <.5] = 0.
            bx.append(image)
            by.append(mask)
            bc.append(get_points(mask, self.input_size))

        if self.only_crd is not None:
            return np.array(bx)[...,np.newaxis], np.array(bc)
        else:
            return np.array(bx)[...,np.newaxis], np.array(by).astype(np.float32), np.array(bc)

    def get_augment(self):
        self.AUG = albu.Compose([
            albu.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
            albu.RandomGamma(gamma_limit=(100,400)),
            albu.GaussNoise(var_limit=(.001, .002)),
            albu.GaussianBlur(),
            # albu.HorizontalFlip(),
            # albu.VerticalFlip(),
            # albu.ElasticTransform(alpha=1, sigma=2, approximate=True, p=.1),
            # albu.GridDistortion(num_steps=1, p=.1)
        ], p=.6)


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


def get_points(mask, nv=None):
    assert mask.shape[-1] == 8

    points = []
    points.extend(get_aortic_point(mask[...,0]))
    points.extend(get_carina_point(mask[...,1]))
    for i in range(2, 8):
        points.extend(get_roi_point(mask[...,i]))

    points = np.array(points).astype(np.float32)
    if nv is not None:
        return points/nv
    else:
        return points


def get_bbox(mask, margin=0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        rects.append(cv2.boundingRect(cnt))

    top_x, top_y, bottom_x, bottom_y = 0,0,0,0
    rects.sort()
    top_x = min([x for (x, y, w, h) in rects]) - margin
    top_y = min([y for (x, y, w, h) in rects]) - margin
    bottom_x = min([x+w for (x, y, w, h) in rects]) + margin
    bottom_y = min([y+h for (x, y, w, h) in rects]) + margin

    mask_size = mask.shape
    if (top_x < 0): top_x = 0
    if (top_y < 0): top_y = 0
    if (bottom_x >= mask_size[1]): bottom_x = mask_size[1] - 1
    if (bottom_y >= mask_size[0]): bottom_y = mask_size[0] - 1

    return (top_x, top_y, bottom_x, bottom_y)


def get_midpoint(mask, y=None):
    if y is None:
        bbox_line = get_bbox(mask)
        if bbox_line is None:
            return
        y = bbox_line[1] + (bbox_line[3] - bbox_line[1]) / 2
        y = int(y)

    LineMask_binary = np.asarray(mask)
    x_candidates1 = []
    y_start = y;
    while np.asarray(x_candidates1).size == 0:
        x_candidates1 = np.where(LineMask_binary[y_start, :] > 0)
        y_start = y_start - 1
        if (y_start < 0):
            break;
    x_candidates2 = []
    y_start = y
    while np.asarray(x_candidates2).size == 0:
        x_candidates2 = np.where(LineMask_binary[y_start, :] > 0)
        y_start = y_start + 1
        if (y_start == LineMask_binary.shape[0]):
            break;

    x_candidates = np.append(x_candidates1, x_candidates2)

    if np.asarray(x_candidates).size > 0:
        x = np.min(x_candidates) + (np.max(x_candidates) - np.min(x_candidates)) / 2
        return (x, y)
    else:
        return


def get_it_midpoint(mask, p1, p2, bl="xy"):
    py = (p1[1] + p2[1]) // 2
    px_cand = np.where(mask[py, :] >0)[0]
    px = px_cand[int((len(px_cand)-1)/2)]
    p1 = [px, py]

    px = (p1[0] + p2[0]) // 2
    py_cand = np.where(mask[:, px] >0)[0]
    py = py_cand[int((len(py_cand)-1)/2)]
    p2 = [px, py]

    if bl == "x": return p2
    elif bl == "y": return p1
    else: return [(p1[0]+p2[0])//2, (p1[1]+p2[1])//2]


### TODO ::: 수정 필요
def get_aortic_point(mask):
    results = []
    p_candidates = np.where(mask > 0)

    p_top_x = p_candidates[1].min()
    p_top_y_candidates = np.where(mask[:, p_top_x] >0)[0]
    p_top_y = p_top_y_candidates[int((len(p_top_y_candidates) - 1) / 2)]
    p_top = [p_top_x, p_top_y]
    results.extend(p_top)

    p_mid = get_midpoint(mask)
    p_mid = [int(p_mid[0]), int(p_mid[1])]

    p_bot_y = p_candidates[0].max()
    p_bot_x_candidates = np.where(mask[p_bot_y, :] >0)[0]
    p_bot_x = p_bot_x_candidates[int((len(p_bot_x_candidates) - 1) / 2)]

    p_top_mid_y = (p_top_y + p_mid[1]) // 2
    p_top_mid_x_candidates = np.where(mask[p_top_mid_y, :] >0)[0]
    p_top_mid_x = p_top_mid_x_candidates[int((len(p_top_mid_x_candidates)-1)/2)]
    p_top_mid_1 = np.array([p_top_mid_x, p_top_mid_y])

    p_top_mid_x = (p_top_x + p_mid[0]) // 2
    p_top_mid_y_candidates = np.where(mask[:, p_top_mid_x] >0)[0]
    p_top_mid_y = p_top_mid_y_candidates[int((len(p_top_mid_y_candidates)-1)/2)]
    p_top_mid_2 = np.array([p_top_mid_x, p_top_mid_y])

    results.extend([int((p_top_mid_1[0]+p_top_mid_2[0])/2),
                    int((p_top_mid_1[1]+p_top_mid_2[1])/2)])

    # add midpoint
    results.extend([p_mid[0], p_mid[1]])

    p_mid_bot_y = (p_bot_y + p_mid[1]) // 2
    p_mid_bot_x_candidates = np.where(mask[p_mid_bot_y, :] >0)[0]
    p_mid_bot_x = p_mid_bot_x_candidates[int((len(p_mid_bot_x_candidates)-1)/2)]
    p_mid_bot_1 = np.array([p_mid_bot_x, p_mid_bot_y])

    p_mid_bot_x = (p_bot_x + p_mid[0]) // 2
    p_mid_bot_y_candidates = np.where(mask[:, p_mid_bot_x] >0)[0]
    p_mid_bot_y = p_mid_bot_y_candidates[int((len(p_mid_bot_y_candidates)-1)/2)]
    p_mid_bot_2 = np.array([p_mid_bot_x, p_mid_bot_y])

    results.extend([int((p_mid_bot_1[0]+p_mid_bot_2[0])/2),
                    int((p_mid_bot_1[1]+p_mid_bot_2[1])/2)])

    # add botpoint
    results.extend([p_bot_x, p_bot_y])
    return results


def get_carina_point(mask):
    results = []
    p_candidates = np.where(mask > 0)

    p_top_y = p_candidates[0].min()
    p_top_x_candidates = np.where(mask[p_top_y, :] > 0)[0]
    p_top_x = p_top_x_candidates[int((len(p_top_x_candidates) - 1) / 2)]
    # p_top = np.array([p_top_x, p_top_y])

    p_left_x = p_candidates[1].min()
    p_left_y_candidates = np.where(mask[:, p_left_x] > 0)[0]
    p_left_y = p_left_y_candidates[int((len(p_left_y_candidates) - 1) / 2)]
    # p_left = np.array([p_left_x, p_left_y])
    results.extend([p_left_x, p_left_y])

    p_right_x = p_candidates[1].max()
    p_right_y_candidates = np.where(mask[:, p_right_x] > 0)[0]
    p_right_y = p_right_y_candidates[int((len(p_right_y_candidates) - 1) / 2)]
    # p_right = np.array([p_right_x, p_right_y])

    p_left_top_x = (p_left_x + p_top_x) // 2
    p_left_top_y_candidates = np.where(mask[:, p_left_top_x] >0)[0]
    p_left_top_y = p_left_top_y_candidates[int((len(p_left_top_y_candidates)-1)/2)]
    p_left_top_1 = np.array([p_left_top_x, p_left_top_y])

    p_left_top_y = (p_left_y + p_top_y) // 2
    p_left_top_x_candidates = np.where(mask[p_left_top_y, :] >0)[0]
    p_left_top_x_candidates = p_left_top_x_candidates[p_left_top_x_candidates < p_top_x]
    p_left_top_x = p_left_top_x_candidates[int((len(p_left_top_x_candidates)-1)/2)]
    p_left_top_2 = np.array([p_left_top_x, p_left_top_y])

    # p_left_top = np.array([int((p_left_top_1[0] + p_left_top_2[0])/2),
    #                        int((p_left_top_1[1] + p_left_top_2[1])/2)])
    results.extend([int((p_left_top_1[0] + p_left_top_2[0])/2),
                    int((p_left_top_1[1] + p_left_top_2[1])/2)])

    # add toppoints
    results.extend([p_top_x, p_top_y])

    p_top_right_x = (p_right_x + p_top_x) // 2
    p_top_right_y_candidates = np.where(mask[:, p_top_right_x] >0)[0]
    p_top_right_y = p_top_right_y_candidates[int((len(p_top_right_y_candidates)-1)/2)]
    p_top_right_1 = np.array([p_top_right_x, p_top_right_y])

    p_top_right_y = (p_right_y + p_top_y) // 2
    p_top_right_x_candidates = np.where(mask[p_top_right_y, :] >0)[0]
    p_top_right_x_candidates = p_top_right_x_candidates[p_top_right_x_candidates> p_top_x]
    p_top_right_x = p_top_right_x_candidates[int((len(p_top_right_x_candidates)-1)/2)]
    p_top_right_2 = np.array([p_top_right_x, p_top_right_y])

    # p_top_right = np.array([int((p_top_right_1[0] + p_top_right_2[0])/2),
    #                         int((p_top_right_1[1] + p_top_right_2[1])/2)])
    results.extend([int((p_top_right_1[0] + p_top_right_2[0])/2),
                    int((p_top_right_1[1] + p_top_right_2[1])/2)])

    # add right point
    results.extend([p_right_x, p_right_y])

    return results


def get_roi_point(mask):
    results = []
    p_candidates = np.where(mask > 0)

    p_top_y = p_candidates[0].min()
    p_top_x_candidates = np.where(mask[p_top_y, :] >0)[0]
    p_top_x = p_top_x_candidates[int((len(p_top_x_candidates) - 1) / 2)]
    p_top = [p_top_x, p_top_y]
    # p_top = np.array([p_top_x, p_top_y])
    results.extend(p_top)

    p_mid = get_midpoint(mask)
    p_mid = np.array([int(p_mid[0]), int(p_mid[1])])

    p_bot_y = p_candidates[0].max()
    p_bot_x_candidates = np.where(mask[p_bot_y, :] >0)[0]
    p_bot_x = p_bot_x_candidates[int((len(p_bot_x_candidates) - 1) / 2)]
    # p_bot = np.array([p_bot_x, p_bot_y])

    p_top_mid_y = (p_top_y + p_mid[1]) // 2
    p_top_mid_x_candidates = np.where(mask[p_top_mid_y, :] >0)[0]
    p_top_mid_x = p_top_mid_x_candidates[int((len(p_top_mid_x_candidates)-1)/2)]
    p_top_mid_1 = np.array([p_top_mid_x, p_top_mid_y])

    # p_top_mid_x = (p_top_x + p_mid[0]) // 2
    # p_top_mid_y_candidates = np.where(mask[:, p_top_mid_x] >0)[0]
    # p_top_mid_y = p_top_mid_y_candidates[int((len(p_top_mid_y_candidates)-1)/2)]
    # p_top_mid_2 = np.array([p_top_mid_x, p_top_mid_y])

    # p_top_mid = np.array([int((p_top_mid_1[0]+p_top_mid_2[0])/2),
    #                       int((p_top_mid_1[1]+p_top_mid_2[1])/2)])
    # results.extend([int((p_top_mid_1[0]+p_top_mid_2[0])/2),
    #                 int((p_top_mid_1[1]+p_top_mid_2[1])/2)])
    results.extend([p_top_mid_1[0], p_top_mid_1[1]])

    # add midpoints
    results.extend([p_mid[0], p_mid[1]])

    p_mid_bot_y = (p_bot_y + p_mid[1]) // 2
    p_mid_bot_x_candidates = np.where(mask[p_mid_bot_y, :] >0)[0]
    p_mid_bot_x = p_mid_bot_x_candidates[int((len(p_mid_bot_x_candidates)-1)/2)]
    p_mid_bot_1 = np.array([p_mid_bot_x, p_mid_bot_y])

    # p_mid_bot_x = (p_bot_x + p_mid[0]) // 2
    # p_mid_bot_y_candidates = np.where(mask[:, p_mid_bot_x] >0)[0]
    # p_mid_bot_y = p_mid_bot_y_candidates[int((len(p_mid_bot_y_candidates)-1)/2)]
    # p_mid_bot_2 = np.array([p_mid_bot_x, p_mid_bot_y])

    # p_mid_bot = np.array([int((p_mid_bot_1[0]+p_mid_bot_2[0])/2),
    #                       int((p_mid_bot_1[1]+p_mid_bot_2[1])/2)])
    # results.extend([int((p_mid_bot_1[0]+p_mid_bot_2[0])/2),
    #                 int((p_mid_bot_1[1]+p_mid_bot_2[1])/2)])
    results.extend([p_mid_bot_1[0], p_mid_bot_1[1]])

    # add bot point
    results.extend([p_bot_x, p_bot_y])

    return results


def draw_line(crd, size, thickness=5, overlay=None):
    crd = crd.reshape(-1, 10)

    sk = np.zeros((size, size, 8))

    for ch in range(8):
        c1 = np.zeros((size, size))
        for i in range(0, len(crd[ch])-2, 2):
            c1 = cv2.line(c1, (crd[ch][i], crd[ch][i+1]), (crd[ch][i+2], crd[ch][i+3]), thickness=thickness, color=1)
        sk[...,ch] = c1

    if overlay is None:
        return sk.astype(np.float32)
    else:
        bg = np.sum(sk, axis=-1)
        bg[bg == 0] = 255.
        bg[bg < 255] = 1.
        bg[bg == 255] = 0.
        return bg.astype(np.float32)

