import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists

from networks import CoEffB4Unetpp
from loader import normalize
from utils import get_patches, merge_patches


def inference():
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/output"
    IMG_SIZE = 768

    W_path = "/data/volume/sinyu/Exp22_1"
    W = sorted(glob(join(W_path, "*")))[-1]

    model = CoEffB4Unetpp((IMG_SIZE, IMG_SIZE, 1), ch=16)
    model.load_weights(W)
    print("Load ", W)

    test_files = glob(join(TEST_DIR, "*"))


    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        img = normalize(pydicom.dcmread(f).pixel_array).astype(np.float32)
        print(CASE_ID, img.shape, "[",i,"/",len(test_files),"]")

        h,w = img.shape
        resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        pred = model.predict(resized_img[np.newaxis, ..., np.newaxis])[-1]
        pred = np.squeeze(pred)
        pred[pred >= (1/9)] = 255.
        pred[pred < (1/9)] = 0.

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


def inference_m():
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/output"
    IMG_SIZE = 768

    W_path = "/data/volume/sinyu/Exp22_1"
    W = sorted(glob(join(W_path, "*")))[-1]

    model = CoEffB4Unetpp((IMG_SIZE, IMG_SIZE, 1), ch=16)
    model.load_weights(W)
    print("Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        img = normalize(pydicom.dcmread(f).pixel_array).astype(np.float32)
        print(CASE_ID, img.shape, "[",i,"/",len(test_files),"]")

        h,w = img.shape
        resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        pred = model.predict(resized_img[np.newaxis, ..., np.newaxis])
        pred = np.squeeze(pred)
        # pred[pred >= (1/9)] = 255.
        # pred[pred < (1/9)] = 0.
        pred = np.argmax(pred, axis=-1)

        pred_assemble = []
        for i in range(8):
            pred_ = pred.copy()
            pred_[pred_ == i] = 255.
            pred_[pred_ < 255] = 0.
            pred_assemble.append(pred_.astype(np.uint8))

        names = [
            "Aortic Knob",
            "Carina",
            "DAO",
            "LAA",
            "Lt Lower CB",
            "Pulmonary Conus",
            "Rt Lower CB",
            "Rt Upper CB"
        ]

        for i, class_name in enumerate(names):
            pred_ = cv2.resize(pred_assemble[i], (w,h))
            pred_[pred_ > 0] = 255
            cv2.imwrite(join(CASE_DIR, CASE_ID+"_"+class_name+".png"), pred_)


if __name__ == "__main__":
    print("Start Inference ...")
    inference_m()
