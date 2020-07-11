import albumentations as albu
from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

from loader import DataLoader_v2
from networks import custom_xception
from utils import CustomCallbacks, bce_dice_loss, iou_metric, bce_dice_focal_loss


def train():
    ### hyper-parameters ================
    N_PATCHES = 16
    PATCH_SIZE = 320

    EPOCHS = 40

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
    LOG_DIR = "/data/volume/Exp03"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader_1 = DataLoader_v2("train", TRAIN_DIR, N_PATCHES, PATCH_SIZE)
    train_loader_2 = DataLoader_v2("train", TRAIN_DIR, N_PATCHES, PATCH_SIZE, AUGMENTATION_TRAIN)
    valid_loader = DataLoader_v2("valid", TRAIN_DIR, N_PATCHES, PATCH_SIZE)

    model = custom_xception((PATCH_SIZE, PATCH_SIZE, 1))

    CB = CustomCallbacks(log_dir=LOG_DIR, nb_epochs=EPOCHS//2, nb_snapshots=1, init_lr=1e-4)

    model.compile(optimizer=Adam(), loss=bce_dice_loss, metrics=['binary_crossentropy', iou_metric])

    model.fit_generator(
        generator=train_loader_1,
        validation_data=valid_loader,
        epochs=EPOCHS//2,
        steps_per_epoch=len(train_loader_1),
        validation_steps=len(valid_loader),
        verbose=1,
        workers=30,
        callbacks=CB.get_callbacks()
    )
    model.save(join(LOG_DIR, "Exp03_1.h5"), save_format="h5")

    model.fit_generator(
        generator=train_loader_2,
        validation_data=valid_loader,
        epochs=EPOCHS//2,
        steps_per_epoch=len(train_loader_2),
        validation_steps=len(valid_loader),
        verbose=1,
        workers=30,
        callbacks=CB.get_callbacks()
    )
    model.save(join(LOG_DIR, "Exp03_2.h5"), save_format="h5")


if __name__ == "__main__":
    print("Start training ...")
    train()
    print("Done !")
