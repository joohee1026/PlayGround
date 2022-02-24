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
    OUTPUT_DIR = "/data/output"
    IMAGE_SIZE = 1024
    LINE_WIDTH = 10
    N_CLASSES = 2*8*5
    print("thick :", LINE_WIDTH)

    # W_path = "/data/volume/sinyu/Exp50_reg"
    # W = sorted(glob(join(W_path, "*")))[-1]
    W = "/data/volume/sinyu/Exp50_reg_v1/121_0.00048_0.01285.h5"
    print("Load ", W)

    model = CoCrdRegressor((IMAGE_SIZE, IMAGE_SIZE, 3), N_CLASSES, "b4")
    model.load_weights(W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        img = get_3ch(pydicom.dcmread(f).pixel_array)
        img = normalize(img).astype(np.float32)

        h,w,_ = img.shape
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        pred = model.predict(resized_img[np.newaxis, ...])
        pred = np.squeeze(pred) * IMAGE_SIZE
        pred = draw_line(pred, IMAGE_SIZE, thickness=LINE_WIDTH)
        pred = pred * 255.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(pred[..., i], (w,h))
            upsample_img[upsample_img < 255] = 0.
            pred_assemble.append(upsample_img.astype(np.uint8))

        for i, class_name in enumerate([
            "Aortic Knob",
            "Carina",
            "DAO",
            "LAA",
            "Lt Lower CB",
            "Pulmonary Conus",
            "Rt Lower CB",
            "Rt Upper CB"
        ]):
            cv2.imwrite(join(CASE_DIR, CASE_ID+"_"+class_name+".png"), pred_assemble[i])


if __name__ == "__main__":
    print("Start Inference ...")
    inference()
