from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import SDataLoader
from networks import EffB4Unetpp_ASPP
from utils import CustomCallbacks
from losses import bce_dice_loss, dice_loss, iou_score


def train():
    ### hyper-parameters ================
    BATCH_SIZE = 2
    IMAGE_SIZE = 768
    VER = "Exp28"
    LR = 1e-3
    CH = 16
    EPOCHS = 150
    CH3 = True
    ### =================================

    TRAIN_DIR = "/data/train"
    LOG_DIR = f"/data/volume/sinyu/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = SDataLoader("train", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE, CH3)
    valid_loader = SDataLoader("valid", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE, CH3)

    model = EffB4Unetpp_ASPP((IMAGE_SIZE, IMAGE_SIZE, 3), ch=CH)
    # model.load_weights("/data/volume/sinyu/Exp27/150_0.29582_0.55889.h5")

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
        initial_epoch=0,
        max_queue_size=30,
        use_multiprocessing=True,
        callbacks=CB.get_callbacks()
    )


def make_output():
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


    TEST_DIR = "/data/test"
    OUTPUT_DIR = "/data/volume/SEGOUT/V28"
    IMAGE_SIZE = 768

    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)
        print("Create path : ", OUTPUT_DIR)

    # W_path = "/data/volume/sinyu/Exp28"
    # W = sorted(glob(join(W_path, "*")))[-1]
    W = "/data/volume/sinyu/Exp28/120_0.30884_0.55158.h5"

    model = EffB4Unetpp_ASPP((IMAGE_SIZE, IMAGE_SIZE, 3), 16)
    model.load_weights(W)
    print("Load ", W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        # CASE_DIR = join(OUTPUT_DIR, CASE_ID)
        #
        # if not exists(CASE_DIR):
        #     makedirs(CASE_DIR)

        img = pydicom.dcmread(f).pixel_array
        img = get_3ch(img)
        img = normalize(img).astype(np.float32)

        h,w,_ = img.shape
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
    print("Start training ...")
    # train()
    make_output()
    print("Done !")
