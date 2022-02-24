import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists
from tensorflow.keras import backend as K

from networks import CoEffB4Unetpp
from loader import normalize, get_3ch, histeq

from regnet import CoCrdRegressor
from segnet import EffB4Unetpp_ASPP


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


def TTA(model, x) :
    _, h, w, c = x.shape
    batch = [x, x[:,::-1,:,:], x[:,:,::-1,:], x[:,::-1,::-1,:]]
    batch = np.stack(batch, axis=0).reshape(-1, h, w, c)

    pb = model.predict(batch)
    pb = np.stack([
        pb[0],
        pb[1][::-1,:,:],
        pb[2][:,::-1,:],
        pb[3][::-1,::-1,:]
    ], axis=0)

    pb = np.squeeze(np.sum(pb, axis=0)) / 4
    pb[pb>=.5] = 255.
    pb[pb<.5] = 0.

    return pb


def regout(LOG_DIR):
    TEST_DIR = "/data/test"
    IMAGE_SIZE = 768
    LINE_WIDTH = 15
    N_CLASSES = 2*8*5

    VER = "EXP51_REG"
    OUTPUT_DIR = join(LOG_DIR, VER)
    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)
        print("Create path : ", OUTPUT_DIR)

    W = "/data/volume/sinyu/Exp50_reg/140_0.00017_0.00943.h5"

    model = CoCrdRegressor((IMAGE_SIZE, IMAGE_SIZE, 1), N_CLASSES, "b4")
    model.load_weights(W)
    print("Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    for f in test_files:
        CASE_ID = f.split("/")[-1][:-4]

        img = pydicom.dcmread(f).pixel_array
        img = histeq(img)
        img = normalize(img).astype(np.float32)

        h, w = img.shape
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img[np.newaxis, ..., np.newaxis]

        pred = model.predict(img)
        pred = np.squeeze(pred) * IMAGE_SIZE
        pred = draw_line(pred, IMAGE_SIZE, thickness=LINE_WIDTH)

        preds = []
        for i in range(8):
            p = cv2.resize(pred[..., i], (w, h))
            p[p > 0] = 1.
            preds.append(p.astype(np.float32))

        output = np.stack(preds, axis=-1)
        np.save(join(OUTPUT_DIR, CASE_ID+".npy"), output)

    return OUTPUT_DIR


def segout(LOG_DIR):
    TEST_DIR = "/data/test"
    IMAGE_SIZE = 768

    VER = "EXP51_SEG"
    OUTPUT_DIR = join(LOG_DIR, VER)
    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)
        print("Create path : ", OUTPUT_DIR)

    W = "/data/volume/sinyu/Exp28/120_0.30884_0.55158.h5"

    model = EffB4Unetpp_ASPP((IMAGE_SIZE, IMAGE_SIZE, 3), 16)
    model.load_weights(W)
    print("Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    for f in test_files:
        CASE_ID = f.split("/")[-1][:-4]

        img = pydicom.dcmread(f).pixel_array
        img = get_3ch(img)
        img = normalize(img).astype(np.float32)

        h, w, _ = img.shape
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img[np.newaxis, ...]

        pred = model.predict(img)
        pred = np.squeeze(pred)
        pred[pred >= .5] = 1.
        pred[pred < .5] = 0.

        preds = []
        for i in range(8):
            p = cv2.resize(pred[..., i], (w, h))
            p[p > 0] = 1.
            preds.append(p.astype(np.float32))

        output = np.stack(preds, axis=-1)
        np.save(join(OUTPUT_DIR, CASE_ID+".npy"), output)

    return OUTPUT_DIR


def inference(REG_DIR=None, SEG_DIR=None):
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/output"
    IMAGE_SIZE = (1152, 896)
    CH = 8

    if REG_DIR is None:
        REG_DIR = "/data/volume/sinyu/TEST/EXP51_REG"
    if SEG_DIR is None:
        SEG_DIR = "/data/volume/sinyu/TEST/EXP51_SEG"

    # W_path = "/data/volume/sinyu/Exp51"
    # W = sorted(glob(join(W_path, "*.h5")))[-1]
    W = "/data/volume/sinyu/Exp51_Eff/010_0.20303_0.69742.h5"

    model = CoEffB4Unetpp((IMAGE_SIZE[0], IMAGE_SIZE[1], 19), ch=CH)
    model.load_weights(W)
    print("Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        image = pydicom.dcmread(f).pixel_array
        image = get_3ch(image)
        image = normalize(image).astype(np.float32)

        seg_mask = np.load(join(SEG_DIR, CASE_ID+".npy"))/1.
        reg_mask = np.load(join(REG_DIR, CASE_ID+".npy"))/1.

        image = np.concatenate([image, seg_mask, reg_mask], axis=-1).astype(np.float32)

        h, w, _ = image.shape
        resized_img = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        resized_img = resized_img[np.newaxis, ...]

        # pred = TTA(model, resized_img)
        pred = model.predict(resized_img)
        pred = np.squeeze(pred)
        pred[pred >= .95] = 1.
        pred[pred < .95] = 0.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(pred[..., i], (w,h))
            upsample_img[upsample_img > 0] = 255.
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
    # VER = "TEST"
    # LOG_DIR = f"/data/volume/sinyu/{VER}"
    #
    # if not exists(LOG_DIR):
    #     makedirs(LOG_DIR)
    #     print("Create path : ", LOG_DIR)
    #
    # print("build reg output")
    # REG_DIR = regout(LOG_DIR)
    #
    # K.clear_session()
    #
    # print("build seg output")
    # SEG_DIR = segout(LOG_DIR)

    REG_DIR = "/data/volume/sinyu/TEST/EXP51_REG"
    SEG_DIR = "/data/volume/sinyu/TEST/EXP51_SEG"

    print("Start Inference ...")
    inference(REG_DIR, SEG_DIR)
