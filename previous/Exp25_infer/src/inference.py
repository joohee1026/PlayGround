import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists
from tensorflow.keras import backend as K

from loader import normalize


# 1. 768 total lr prediction
# 2. cropping with 50~100 pad
# 3. roi prediction (sigmoid)

def get_roi(y):
    nonzero = np.argwhere(y)
    (e1, e2), (e3, e4) = nonzero.min(axis=0), nonzero.max(axis=0)
    return (e1, e2), (e3, e4)


def tst_infer():
    from exp27_networks import CoEffB4Unetpp_ASPP

    TEST_DIR = "/data/test"
    IMAGE_SIZE = 512

    model = CoEffB4Unetpp_ASPP((IMAGE_SIZE, IMAGE_SIZE, 1), 16)
    W = "/data/volume/sinyu/Exp27/150_0.29582_0.55889.h5"
    model.load_weights(W)
    print("--> 1 step load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    roi_data = {}
    for f in test_files:
        CASE_ID = f.split("/")[-1][:-4]

        img = pydicom.dcmread(f).pixel_array
        img = cv2.createCLAHE(tileGridSize=(3, 3)).apply(img)
        img = normalize(img).astype(np.float32)

        h, w = img.shape
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        resized_img = resized_img[np.newaxis, ..., np.newaxis]

        pred = model.predict(resized_img)
        pred = np.squeeze(pred)
        pred[pred >= .5] = 1.
        pred[pred < .5] = 0.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(pred[..., i], (w,h))
            upsample_img[upsample_img > 0] = 1.
            pred_assemble.append(upsample_img.astype(np.uint8))

        pred = np.stack(pred_assemble, axis=-1)

        # pred = np.array(pred_assemble)
        bg = np.sum(pred, axis=-1)
        bg[bg >0] = 1
        # bg = 1 - bg_
        # bg += 1
        # bg[bg >1] = 0

        (e1, e2), (e3, e4) = get_roi(bg)
        roi_data[CASE_ID] = [img, [e1,e2,e3,e4]]

    return roi_data


def tst_infer2(roi_data, pad=50):
    from networks import CoEffB4Unetpp_ASPP

    OUTPUT_DIR = "/data/output"

    IMAGE_SIZE = 512
    CH = 16
    model = CoEffB4Unetpp_ASPP((IMAGE_SIZE,IMAGE_SIZE,1), ch=CH)
    W = "/data/volume/sinyu/Exp25_1/130_0.29967_0.56959.h5"
    model.load_weights(W)
    print("--> 2 step load ", W)

    for CASE_ID in list(roi_data.keys()):
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        (img, e_lst) = roi_data[CASE_ID]
        (e1, e2, e3, e4) = e_lst
        ad1, ad2, ad3, ad4 = (e1-pad), (e2-pad), (img.shape[0]-e3-pad), (img.shape[1]-e4-pad)
        roi = img[e1-pad:e3+pad, e2-pad:e4+pad]
        h, w = roi.shape

        roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
        roi = roi[np.newaxis, ..., np.newaxis]

        roi_pred = model.predict(roi)
        roi_pred = np.squeeze(roi_pred)
        roi_pred[roi_pred >= .5] = 255.
        roi_pred[roi_pred < .5] = 0.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(roi_pred[...,i], (w,h))
            upsample_img[upsample_img > 0] = 255.
            upsample_img = np.pad(upsample_img, ((ad1, ad3), (ad2, ad4)), "constant")
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
    roi_data = tst_infer()

    print(" ... ", roi_data[list(roi_data.keys())[0]][0].shape)
    print(" ... ", roi_data[list(roi_data.keys())[0]][1])

    print("stage 1 done.")
    K.clear_session()

    tst_infer2(roi_data, 16)
    print("stage 2 done.")
