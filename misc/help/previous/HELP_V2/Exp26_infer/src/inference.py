import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists

from networks import CoEffB4Unetpp
from loader import normalize


def inference():
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/output"
    IMAGE_SIZE = 768
    CH = 16

    # W_path = "/data/volume/sinyu/Exp26"
    # W = sorted(glob(join(W_path, "*")))[-1]
    W = "/data/volume/sinyu/Exp26/110_0.29944_0.55617.h5"

    model = CoEffB4Unetpp((IMAGE_SIZE, IMAGE_SIZE, 1), ch=CH, module="cbam", n_classes=8)
    model.load_weights(W)
    print("Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        img = pydicom.dcmread(f).pixel_array
        img = cv2.createCLAHE(tileGridSize=(3,3)).apply(img)
        img = normalize(img).astype(np.float32)

        # print(CASE_ID, img.shape, "[",i,"/",len(test_files),"]")

        h,w = img.shape
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

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
