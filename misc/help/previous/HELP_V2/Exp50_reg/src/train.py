from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import CrdSegSDataLoader
from networks import CoCrdRegressor
from utils import CustomCallbacks
from losses import bce_dice_loss, dice_loss, iou_score


def regressor():
    BATCH_SIZE = 3
    IMAGE_SIZE = 1024
    VER = "Exp50_reg_v1"
    LR = 1e-3
    EPOCHS = 200
    N_CLASSES = 2*8*5

    TRAIN_DIR = "/data/train"
    LOG_DIR = f"/data/volume/sinyu/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = CrdSegSDataLoader("train", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE, True, only_crd=True)
    valid_loader = CrdSegSDataLoader("valid", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE, True, only_crd=True)

    model = CoCrdRegressor((IMAGE_SIZE, IMAGE_SIZE, 3), N_CLASSES, "b4")
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
        callbacks=CB.get_callbacks(mode="reg")
    )


if __name__ == "__main__":
    print("Start training ...")
    regressor()
    print("Done !")
