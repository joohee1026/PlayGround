import albumentations as albu
from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import DataLoader
from networks import custom_B4
from utils import CustomCallbacks
from losses import cce_dice_loss, iou_, dice_, focal_, iou_metric


def train():
    ### hyper-parameters ================
    N_PATCHES = 8
    PATCH_SIZE = 512
    VER = "Exp07"
    LR = 1e-3

    EPOCHS = 200
    AUGMENTATION_TRAIN = albu.Compose([
        # albu.ElasticTransform(alpha=.5, sigma=1, approximate=True, p=.1),
        albu.RandomBrightnessContrast(p=1),
        albu.RandomGamma(p=1),
        albu.ShiftScaleRotate(),
        albu.GaussNoise(var_limit=(.001, .002)),
        albu.GaussianBlur(),
        albu.Blur(blur_limit=3),
        albu.MotionBlur(blur_limit=3),
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.RandomRotate90(),
        # albu.RandomSizedCrop(min_max_height=(int(PATCH_SIZE//1.5), PATCH_SIZE), height=PATCH_SIZE, width=PATCH_SIZE, p=.25),
    ], p=1)

    ### =================================

    TRAIN_DIR = "/data/train"
    LOG_DIR = f"/data/volume/sinyu/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = DataLoader("train", TRAIN_DIR, False, N_PATCHES, PATCH_SIZE, AUGMENTATION_TRAIN)
    valid_loader = DataLoader("valid", TRAIN_DIR, False, 5, PATCH_SIZE)

    model = custom_B4((None, None, 1))
    model.load_weights("/data/volume/sinyu/Exp07/100_0.37328_0.53043.h5")


    CB = CustomCallbacks(log_dir=LOG_DIR, nb_epochs=EPOCHS, nb_snapshots=1, init_lr=LR)

    model.compile(
        optimizer=Adam(),
        loss=cce_dice_loss,
        metrics=['categorical_crossentropy', dice_, focal_, iou_, iou_metric]
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
        callbacks=CB.get_callbacks(),
        initial_epoch=100
    )
    model.save(join(LOG_DIR, f"{VER}_f.h5"), save_format="h5")


if __name__ == "__main__":
    print("Start training ...")
    train()
    print("Done !")
