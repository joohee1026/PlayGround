from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import DataLoader
from networks import CoEffB4Unetpp
from utils import CustomCallbacks
from losses import cce_dice_loss, dice_loss, iou_score


def train():
    ### hyper-parameters ================
    BATCH_SIZE = 3
    IMAGE_SIZE = 768
    VER = "Exp22_1"
    LR = 1e-3
    CH = 16
    EPOCHS = 100

    ### =================================

    TRAIN_DIR = "/data/train"
    LOG_DIR = f"/data/volume/sinyu/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = DataLoader("train", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE)
    valid_loader = DataLoader("valid", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE)

    model = CoEffB4Unetpp((IMAGE_SIZE, IMAGE_SIZE, 1), ch=CH)
    model.load_weights("/data/volume/sinyu/Exp22/023_0.33226_0.56584.h5")

    CB = CustomCallbacks(log_dir=LOG_DIR, nb_epochs=EPOCHS, nb_snapshots=1, init_lr=LR)

    model.compile(
        optimizer=Adam(),
        loss=cce_dice_loss,
        metrics=['categorical_crossentropy', dice_loss, iou_score]
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
    # model.save(join(LOG_DIR, f"{VER}_f.h5"), save_format="h5")


if __name__ == "__main__":
    print("Start training ...")
    train()
    print("Done !")
