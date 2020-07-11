import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists

from networks import EffB4Unetpp_v2
from loader import normalize, random_crop
from utils import get_patches, merge_patches


def img_prep(f_path, v1=None, v2=None, v3=None, v4=None, img_size=512):
    img = pydicom.dcmread(f_path).pixel_array
    img = img[v1:-v2, v3:-v4]
    orig_h, orig_w = img.shape
    img = normalize(cv2.resize(img, (img_size, img_size))).astype(np.float32)
    return img, orig_h, orig_w


def inference():
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/output"
    IMAGE_SIZE = 512

    W_path = "/data/volume/sinyu/Exp19"
    W = sorted(glob(join(W_path, "*")))[-1]

    model = EffB4Unetpp_v2((IMAGE_SIZE, IMAGE_SIZE, 1), ch=16)
    model.load_weights(W)
    print("Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        # img = normalize(pydicom.dcmread(f).pixel_array).astype(np.float32)
        resized_img, h, w = img_prep(f, 100,100,100,100, IMAGE_SIZE)
        print(CASE_ID, (h,w), "[",i,"/",len(test_files),"]")

        # h,w = img.shape
        # resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        pred = model.predict(resized_img[np.newaxis, ..., np.newaxis])
        pred = np.squeeze(pred)
        pred[pred >= .5] = 255.
        pred[pred < .5] = 0.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(pred[..., i], (w,h))
            upsample_img = np.pad(upsample_img, ((100,100),(100,100)), "constant")
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
    print("Start Inference ...")
    inference()
