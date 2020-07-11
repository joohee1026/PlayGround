from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import CrdSegSDataLoader
from networks import CoCrdRegressor
from utils import CustomCallbacks


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


if __name__ == "__main__":
    print("Start training ...")
    # regressor()
    print("Done !")
