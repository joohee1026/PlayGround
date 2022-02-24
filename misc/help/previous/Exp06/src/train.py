import albumentations as albu
from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import DataLoader_lr
from networks import custom_B4
from utils import CustomCallbacks
from losses import iou_metric, cce_dice_loss, _dice_loss


def train():
    ### hyper-parameters ================
    N_PATCHES = 8
    PATCH_SIZE = 512
    VER = "Exp06"

    EPOCHS = 100
    NONA_EP = 0
    AUGMENTATION_TRAIN = albu.Compose([
        # albu.CLAHE(),
        albu.ElasticTransform(alpha=.5, sigma=1, approximate=True, p=.1),
        albu.RandomBrightnessContrast(p=1),
        albu.RandomGamma(p=1),
        albu.GaussNoise(var_limit=(.001, .002)),
        albu.GaussianBlur(),
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.RandomRotate90(),
        # albu.RandomSizedCrop(min_max_height=(PATCH_SIZE//2, PATCH_SIZE), height=PATCH_SIZE, width=PATCH_SIZE, p=.25),
    ], p=1)

    ### =================================

    TRAIN_DIR = "/data/train"
    LOG_DIR = f"/data/volume/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    # train_loader_1 = DataLoader_lr("train", TRAIN_DIR, N_PATCHES, PATCH_SIZE)
    train_loader_2 = DataLoader_lr("train", TRAIN_DIR, N_PATCHES, PATCH_SIZE, AUGMENTATION_TRAIN)
    valid_loader = DataLoader_lr("valid", TRAIN_DIR, N_PATCHES, PATCH_SIZE)

    model = custom_B4((None, None, 1))
    # model.load_weights(join(LOG_DIR, "tmp2.h5"))


    CB = CustomCallbacks(log_dir=LOG_DIR, nb_epochs=EPOCHS, nb_snapshots=1, init_lr=1e-3)

    model.compile(
        optimizer=Adam(),
        loss=cce_dice_loss,
        metrics=['categorical_crossentropy', _dice_loss, iou_metric]
    )

    model.fit_generator(
        generator=train_loader_2,
        validation_data=valid_loader,
        epochs=(EPOCHS-NONA_EP),
        steps_per_epoch=len(train_loader_2),
        validation_steps=len(valid_loader),
        verbose=1,
        workers=30,
        callbacks=CB.get_callbacks()
    )
    model.save(join(LOG_DIR, f"{VER}_2.h5"), save_format="h5")


if __name__ == "__main__":
    print("Start training ...")
    train()
    print("Done !")
