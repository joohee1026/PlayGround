import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists

from networks import DeepLabV3p_Xc
from loader import normalize
from utils import get_patches, merge_patches


def inference():
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/output"
    PATCH_TRAIN = False
    IMG_SIZE = 512

    W_path = "/data/volume/sinyu/Exp14"
    W = sorted(glob(join(W_path, "*")))[-1]

    model = DeepLabV3p_Xc((None, None, 1))
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

        if PATCH_TRAIN:
            img = img[..., np.newaxis]
            patches, cores, orig_shape, b = get_patches(img, patch_size=(IMG_SIZE-32))
            seg_patches = []
            for patch in patches:
                pred = model.predict(patch[np.newaxis,...])
                seg_patches.append(np.squeeze(pred))

            pred_assemble = merge_patches(seg_patches, cores, orig_shape, b)
            pred_assemble[pred_assemble >=.5] = 255.
            pred_assemble[pred_assemble < .5] = 0.
            pred_assemble = pred_assemble.astype(np.uint8)
        else:
            h,w = img.shape
            resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            pred = model.predict(resized_img[np.newaxis, ..., np.newaxis])
            pred = np.squeeze(pred)
            pred[pred >= .5] = 255.
            pred[pred < .5] = 0.

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
    print("Start Inference ...")
    inference()
