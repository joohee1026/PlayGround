from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import CrdSegSDataLoader
from networks import CoCrdRegressor
from utils import CustomCallbacks
from losses import bce_dice_loss, dice_loss, iou_score


def regressor():
    BATCH_SIZE = 5
    IMAGE_SIZE = 768
    VER = "Exp50_reg_B5"
    LR = 1e-3
    EPOCHS = 150

    TRAIN_DIR = "/data/train"
    LOG_DIR = f"/data/volume/sinyu/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = CrdSegSDataLoader("train", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE, only_crd=True)
    valid_loader = CrdSegSDataLoader("valid", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE, only_crd=True)

    model = CoCrdRegressor((IMAGE_SIZE, IMAGE_SIZE, 1), 80, "b5")
    model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["mae"])

    CB = CustomCallbacks(log_dir=LOG_DIR, nb_epochs=EPOCHS, nb_snapshots=1, init_lr=LR)

    model.fit_generator(
        generator=train_loader,
        validation_data=valid_loader,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        validation_steps=len(valid_loader),
        verbose=1,
        workers=6,
        max_queue_size=30,
        use_multiprocessing=True,
        callbacks=CB.get_callbacks(mode="clf")
    )



def make_output():
    import cv2
    import pydicom
    import numpy as np
    from glob import glob
    from os import makedirs
    from os.path import join, exists

    from networks import CoCrdRegressor
    from loader import normalize, histeq, get_3ch


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


    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/volume/REGOUT/V0"
    IMAGE_SIZE = 768
    LINE_WIDTH = 20
    N_CLASSES = 2*8*5
    print("thick :", LINE_WIDTH)

    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)
        print("Create path : ", OUTPUT_DIR)

    # W_path = "/data/volume/sinyu/Exp50_reg"
    # W = sorted(glob(join(W_path, "*")))[-1]
    W = "/data/volume/sinyu/Exp50_reg/140_0.00017_0.v00943.h5"
    print("Load ", W)

    model = CoCrdRegressor((IMAGE_SIZE, IMAGE_SIZE, 1), N_CLASSES, "b4")
    model.load_weights(W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        # CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        # if not exists(CASE_DIR):
        #     makedirs(CASE_DIR)

        img = histeq(pydicom.dcmread(f).pixel_array)
        # img = get_3ch(pydicom.dcmread(f).pixel_array)
        img = normalize(img).astype(np.float32)
        # print(CASE_ID, img.shape, "[",i,"/",len(test_files),"]")

        # h,w,_ = img.shape
        h,w = img.shape
        resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # pred = model.predict(resized_img[np.newaxis, ...])
        pred = model.predict(resized_img[np.newaxis, ..., np.newaxis])
        pred = np.squeeze(pred) * IMAGE_SIZE
        pred = draw_line(pred, IMAGE_SIZE, thickness=LINE_WIDTH)
        pred = pred * 255.

        pred_assemble = []
        for i in range(8):
            upsample_img = cv2.resize(pred[..., i], (w,h))
            upsample_img[upsample_img > 0] = 255.
            pred_assemble.append(upsample_img.astype(np.uint8))

        output = np.stack(pred_assemble, axis=-1)
        np.save(join(OUTPUT_DIR, CASE_ID+".npy"), output)


if __name__ == "__main__":
    print("Start training ...")
    # regressor()
    make_output()
    print("Done !")
