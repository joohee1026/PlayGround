from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import SDataLoader
from networks import CoEffB4Unetpp_ASPP
from utils import CustomCallbacks, get_callbacks
from losses import bce_dice_loss, dice_loss, iou_score


def train():
    ### hyper-parameters ================
    BATCH_SIZE = 5
    IMAGE_SIZE = 512
    VER = "Exp27"
    LR = 1e-5
    CH = 16
    EPOCHS = 150

    ### =================================

    TRAIN_DIR = "/data/train"
    LOG_DIR = f"/data/volume/sinyu/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = SDataLoader("train", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE)
    valid_loader = SDataLoader("valid", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE)

    model = CoEffB4Unetpp_ASPP((IMAGE_SIZE, IMAGE_SIZE, 1), ch=CH)
    model.load_weights("/data/volume/sinyu/Exp27/077_0.30011_0.55439.h5")

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
        initial_epoch=77,
        max_queue_size=30,
        use_multiprocessing=True,
        callbacks=CB.get_callbacks()
        # callbacks=get_callbacks(LOG_DIR)
    )


if __name__ == "__main__":
    print("Start training ...")
    # train()
    print("Done !")