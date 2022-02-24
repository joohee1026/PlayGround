import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists

from networks import CoCrdRegressor
from loader import normalize, histeq, get_3ch


def draw_line(crd, size, thickness=5, overlay=None):
    crd = crd.reshape(8, -1)
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


def inference():
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/volume/REGOUT/EXP50_V1"
    IMAGE_SIZE = 768
    LINE_WIDTH = 20
    N_CLASSES = 2*8*5
    print("thick :", LINE_WIDTH)

    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)
        print("Create :", OUTPUT_DIR)

    # W_path = "/data/volume/sinyu/Exp50_reg"
    # W = sorted(glob(join(W_path, "*")))[-1]
    W = "/data/volume/sinyu/Exp50_reg/140_0.00017_0.00943.h5"
    print("Load ", W)

    model = CoCrdRegressor((IMAGE_SIZE, IMAGE_SIZE, 1), N_CLASSES, "b4")
    model.load_weights(W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]

        img = histeq(pydicom.dcmread(f).pixel_array)
        img = normalize(img).astype(np.float32)

        h,w = img.shape
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        resized_img = resized_img[np.newaxis, ..., np.newaxis]

        pred = model.predict(resized_img)
        pred = np.squeeze(pred) * IMAGE_SIZE
        pred = draw_line(pred, IMAGE_SIZE, thickness=LINE_WIDTH)
        pred = pred * 255.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(pred[..., i], (w,h))
            upsample_img[upsample_img > 0] = 255.
            pred_assemble.append(upsample_img.astype(np.uint8))

        output = np.stack(pred_assemble, axis=-1)
        np.save(join(OUTPUT_DIR, CASE_ID+".npy"), output)


if __name__ == "__main__":
    print("Start Inference ...")
    inference()
