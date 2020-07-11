import cv2
import pydicom
import numpy as np
from os.path import join, exists
from glob import glob
from os import makedirs
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from loader import S3DataLoader, histeq, get_3ch, normalize
from networks import CoEffB4Unetpp
from utils import CustomCallbacks
from losses import bce_dice_loss, dice_loss, iou_score

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


def regout(LOG_DIR):
    TRAIN_DIR = "/data/train"
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

    TRAIN_DIRS = glob(join(TRAIN_DIR, "*"))
    train_files = [glob(join(d, "*.dcm"))[0] for d in TRAIN_DIRS]

    for f in train_files:
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
    TRAIN_DIR = "/data/train"
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

    TRAIN_DIRS = glob(join(TRAIN_DIR, "*"))
    train_files = [glob(join(d, "*.dcm"))[0] for d in TRAIN_DIRS]

    for f in train_files:
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


def train(REG_DIR=None, SEG_DIR=None):
    ### hyper-parameters ================
    BATCH_SIZE = 2
    IMAGE_SIZE = (1152, 896)
    VER = "Exp51_Eff"
    LR = 1e-5
    CH = 8
    EPOCHS = 50
    CH3 = True
    ### =================================

    TRAIN_DIR = "/data/train"
    if REG_DIR is None:
        REG_DIR = "/data/volume/sinyu/TRAIN/EXP51_REG"
    if SEG_DIR is None:
        SEG_DIR = "/data/volume/sinyu/TRAIN/EXP51_SEG"

    LOG_DIR = f"/data/volume/sinyu/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = S3DataLoader("train", TRAIN_DIR, SEG_DIR, REG_DIR, BATCH_SIZE, IMAGE_SIZE, CH3)
    valid_loader = S3DataLoader("valid", TRAIN_DIR, SEG_DIR, REG_DIR, BATCH_SIZE, IMAGE_SIZE, CH3)

    model = CoEffB4Unetpp((IMAGE_SIZE[0], IMAGE_SIZE[1], 19), ch=CH)
    model.load_weights("/data/volume/sinyu/Exp51_Eff/006_0.19143_0.69794.h5")

    CB = CustomCallbacks(log_dir=LOG_DIR, nb_epochs=EPOCHS, nb_snapshots=1, init_lr=LR)

    model.compile(
        optimizer=Adam(),
        loss=bce_dice_loss,
        metrics=['binary_crossentropy', dice_loss, iou_score]
    )

    model.fit_generator(
        generator=train_loader,
        validation_data=valid_loader,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        validation_steps=len(valid_loader),
        verbose=1,
        workers=6,
        initial_epoch=6,
        max_queue_size=30,
        use_multiprocessing=True,
        callbacks=CB.get_callbacks()
    )


if __name__ == "__main__":
    # VER = "TRAIN"
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

    REG_DIR = "/data/volume/sinyu/TRAIN/EXP51_REG"
    SEG_DIR = "/data/volume/sinyu/TRAIN/EXP51_SEG"

    print("Start training ...")
    # train(REG_DIR, SEG_DIR)
    print("Done !")
