import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists

from networks import EffB4Unetpp_ASPP
from loader import normalize, get_3ch


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


def inference():
    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/volume/SEGOUT/EXP50_V1"
    IMAGE_SIZE = 768

    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)
        print("Create :", OUTPUT_DIR)

    # W_path = "/data/volume/sinyu/Exp28"
    # W = sorted(glob(join(W_path, "*")))[-1]
    W = "/data/volume/sinyu/Exp28/120_0.30884_0.55158.h5"

    model = EffB4Unetpp_ASPP((IMAGE_SIZE, IMAGE_SIZE, 3), 16)
    model.load_weights(W)
    print("Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]

        img = pydicom.dcmread(f).pixel_array
        img = get_3ch(img)
        img = normalize(img).astype(np.float32)

        h, w, _ = img.shape
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        resized_img = resized_img[np.newaxis, ...]

        # pred = TTA(model, resized_img)
        pred = model.predict(resized_img)
        pred = np.squeeze(pred)
        pred[pred >= .5] = 255.
        pred[pred < .5] = 0.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(pred[..., i], (w,h))
            upsample_img[upsample_img > 0] = 255.
            pred_assemble.append(upsample_img.astype(np.uint8))

        output = np.stack(pred_assemble, axis=-1)
        np.save(join(OUTPUT_DIR, CASE_ID+".npy"), output)


if __name__ == "__main__":
    print("Start Inference ...")
    inference()
