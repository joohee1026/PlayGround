import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists

from reg_net import CoCrdRegressor
from seg_net import EffB4Unetpp_ASPP
from loader import normalize, histeq, get_3ch

from tensorflow.keras import backend as K


def TTA(model, x) :
    _, h, w, c = x.shape
    batch = [x, x[:,::-1,:,:], x[:,:,::-1,:], x[:,::-1,::-1,:]]
    batch = np.stack(batch, axis=0).reshape(-1,h,w,c)

    pb = model.predict(batch)
    pb = np.stack([
        pb[0],
        pb[1][::-1,:,:],
        pb[2][:,::-1,:],
        pb[3][::-1,::-1,:]
    ], axis=0)

    pb = np.squeeze(np.sum(pb, axis=0)) / 4
    pb[pb>=(1/9)] = 255.
    pb[pb<(1/9)] = 0.

    return pb


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


def build_reg():
    TEST_DIR = "/data/test"
    IMAGE_SIZE = 768
    LINE_WIDTH = 15
    N_CLASSES = 2*8*5
    print("thick : ", LINE_WIDTH)

    OUTPUT_DIR = "/data/volume/REGOUT/EXP50_V1"

    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)
        print("Create : ", OUTPUT_DIR)

    W = "/data/volume/sinyu/Exp50_reg/140_0.00017_0.00943.h5"
    model = CoCrdRegressor((IMAGE_SIZE, IMAGE_SIZE, 1), N_CLASSES, "b4")
    model.load_weights(W)
    print("--> Reg W Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    for f in test_files:
        CASE_ID = f.split("/")[-1][:-4]

        img = histeq(pydicom.dcmread(f).pixel_array)
        img = normalize(img).astype(np.float32)

        h, w = img.shape
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        resized_img = resized_img[np.newaxis, ..., np.newaxis]

        pred = model.predict(resized_img)
        pred = np.squeeze(pred) * IMAGE_SIZE
        pred = draw_line(pred, IMAGE_SIZE, thickness=LINE_WIDTH)
        pred = pred * 1.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(pred[..., i], (w, h))
            upsample_img[upsample_img > 0] = 1.
            pred_assemble.append(upsample_img)

        output = np.stack(pred_assemble, axis=-1)
        np.save(join(OUTPUT_DIR, CASE_ID+".npy"), output)

    return OUTPUT_DIR


def build_seg():
    TEST_DIR = "/data/test"
    IMAGE_SIZE = 768

    OUTPUT_DIR = "/data/volume/SEGOUT/EXP50_V1"

    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)
        print("Create : ", OUTPUT_DIR)

    W = "/data/volume/sinyu/Exp28/120_0.30884_0.55158.h5"
    model = EffB4Unetpp_ASPP((IMAGE_SIZE, IMAGE_SIZE, 3), 16)
    model.load_weights(W)
    print("--> Seg W Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    results = {}
    for f in test_files:
        CASE_ID = f.split("/")[-1][:-4]

        img = pydicom.dcmread(f).pixel_array
        img = get_3ch(img)
        img = normalize(img).astype(np.float32)

        h, w, _ = img.shape
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        resized_img = resized_img[np.newaxis, ...]

        pred = model.predict(resized_img)
        pred = np.squeeze(pred)
        pred[pred >= .5] = 1.
        pred[pred < .5] = 0.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(pred[..., i], (w, h))
            upsample_img[upsample_img > 0] = 1.
            pred_assemble.append(upsample_img)

        output = np.stack(pred_assemble, axis=-1)
        np.save(join(OUTPUT_DIR, CASE_ID+".npy"), output)

    return OUTPUT_DIR


def infer_sum(SEG_DIR=None, REG_DIR=None):
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/output"

    if SEG_DIR is None:
        SEG_DIR = "/data/volume/SEGOUT/EXP50_V1"

    if REG_DIR is None:
        REG_DIR = "/data/volume/REGOUT/EXP50_V1"

    seg_paths = sorted(glob(join(SEG_DIR, "*")))
    reg_paths = sorted(glob(join(REG_DIR, "*")))

    for s, r in zip(seg_paths, reg_paths):
        CASE_ID = s.split("/")[-1][:-4]
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        seg = np.load(s)
        reg = np.load(r)

        output = seg + reg
        output[output > 0] = 255.

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
            cv2.imwrite(join(CASE_DIR, CASE_ID+"_"+class_name+".png"), output[...,i])



if __name__ == "__main__":
    print("reg processing")
    reg_dir = build_reg()

    K.clear_session()

    print("seg processing")
    seg_dir = build_seg()

    print("summation")
    infer_sum(seg_dir, reg_dir)
