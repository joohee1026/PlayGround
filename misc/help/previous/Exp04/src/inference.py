import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists

from networks import custom_B4
from loader import normalize
from utils import get_patches, merge_patches, bce_dice_loss, iou_metric


def inference():
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/output"

    W_path = "/data/volume/Exp04"
    W = sorted(glob(join(W_path, "*")))[-1]
    print("Load ", W)

    model = custom_B4((None, None, 1))
    model.load_weights(W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        img = normalize(pydicom.dcmread(f).pixel_array).astype(np.float32)
        img = img[..., np.newaxis]
        print(CASE_ID, img.shape, "[",i,"/",len(test_files),"]")

        patches, cores, orig_shape, b = get_patches(img, patch_size=480)
        seg_patches = []
        for patch in patches:
            pred = model.predict(patch[np.newaxis,...])
            seg_patches.append(np.squeeze(pred))

        pred_assemble = merge_patches(seg_patches, cores, orig_shape, b)
        pred_assemble[pred_assemble >=.5] = 255.
        pred_assemble[pred_assemble < .5] = 0.

        pred_assemble = pred_assemble.astype(np.uint8)

        for i, class_name in enumerate([
            "Aortic Knob",
            "Carina",
            "DAO",
            "LAA",
            "Lt Lower C8",
            "Pulmonary Conus",
            "Rt Lower CB",
            "Rt Upper CB"
        ]):
            cv2.imwrite(join(CASE_DIR, CASE_ID+"_"+class_name+".png"), pred_assemble[...,i])


if __name__ == "__main__":
    print("Start Inference ...")
    inference()
