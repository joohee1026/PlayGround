from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import SDataLoader
from networks import CoEffB4Unetpp_ASPP
from utils import CustomCallbacks
from losses import bce_dice_loss, dice_loss, iou_score


def train():
    ### hyper-parameters ================
    BATCH_SIZE = 2
    IMAGE_SIZE = 768
    VER = "Exp30"
    LR = 1e-4
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

    model = CoEffB4Unetpp_ASPP((IMAGE_SIZE, IMAGE_SIZE, 3), ch=CH)

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
        max_queue_size=30,
        use_multiprocessing=True,
        callbacks=CB.get_callbacks()
    )


if __name__ == "__main__":
    print("Start training ...")
    train()
    print("Done !")
