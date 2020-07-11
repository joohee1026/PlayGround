import albumentations as albu
from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

from loader import DataLoader_v2
from networks import custom_B4
from utils import CustomCallbacks, bce_dice_loss, iou_metric


def train():
    ### hyper-parameters ================
    N_PATCHES = 16
    PATCH_SIZE = 320

    EPOCHS = 50
    VALID_RATIO = .02

    AUGMENTATION_TRAIN = albu.Compose([
        # albu.CLAHE(),
        albu.RandomBrightnessContrast(p=1),
        albu.RandomGamma(p=1),
        albu.GaussNoise(var_limit=(.001, .005)),
        albu.GaussianBlur(),
        albu.ElasticTransform(approximate=True),
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.RandomRotate90(),
        albu.RandomSizedCrop(min_max_height=(PATCH_SIZE//2, PATCH_SIZE), height=PATCH_SIZE, width=PATCH_SIZE, p=.25),
    ], p=1)

    ### =================================

    TRAIN_DIR = "/data/train"
    LOG_DIR = "/data/volume/Exp01"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = DataLoader_v2("train", TRAIN_DIR, VALID_RATIO, N_PATCHES, PATCH_SIZE, AUGMENTATION_TRAIN)
    valid_loader = DataLoader_v2("valid", TRAIN_DIR, VALID_RATIO, N_PATCHES, PATCH_SIZE)

    model = custom_B4((PATCH_SIZE, PATCH_SIZE, 1))
    CB = CustomCallbacks(log_dir=LOG_DIR, nb_epochs=EPOCHS, nb_snapshots=30, init_lr=1e-3)

    model.compile(optimizer=Adam(), loss=bce_dice_loss, metrics=['binary_crossentropy', iou_metric])

    model.fit_generator(
        generator=train_loader,
        validation_data=valid_loader,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        validation_steps=len(valid_loader),
        verbose=1,
        workers=30,
        callbacks=CB.get_callbacks()
    )
    model.save(join(LOG_DIR, "b4_v1_f.h5"), save_format="h5")


if __name__ == "__main__":
    print("Start training ...")
    train()
    print("Done !")
