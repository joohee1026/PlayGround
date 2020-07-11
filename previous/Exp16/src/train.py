import albumentations as albu
from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import DataLoader_RoI
from networks import EffB4Unetpp_v2
from utils import CustomCallbacks
from losses import cce_dice_loss, dice_loss, iou_score


def train():
    ### hyper-parameters ================
    BATCH_SIZE = 7
    PATCH_SIZE = 512
    VER = "Exp16"
    LR = 1e-4
    CH = 16
    EPOCHS = 100

    AUGMENTATION_TRAIN = albu.Compose([
        # albu.CLAHE(),
        albu.ElasticTransform(alpha=.5, sigma=1, approximate=True, p=.2),
        albu.RandomBrightnessContrast(p=.8),
        albu.RandomGamma(p=.8),
        albu.GaussNoise(var_limit=(.001, .002)),
        albu.GaussianBlur(),
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        # albu.RandomRotate90(),
        # albu.RandomSizedCrop(min_max_height=(PATCH_SIZE//2, PATCH_SIZE), height=PATCH_SIZE, width=PATCH_SIZE, p=.25),
    ], p=1)

    ### =================================

    # TRAIN_DIR = "/data/train"
    TRAIN_DIR = "/data/volume/Datasets/ROI200/TRAIN"
    LOG_DIR = f"/data/volume/sinyu/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = DataLoader_RoI("train", TRAIN_DIR, batch_size=BATCH_SIZE, img_size=PATCH_SIZE, augmentation=AUGMENTATION_TRAIN)
    valid_loader = DataLoader_RoI("valid", TRAIN_DIR, batch_size=BATCH_SIZE, img_size=PATCH_SIZE)

    # train_loader = DataLoader("train", TRAIN_DIR, False, N_PATCHES, PATCH_SIZE, AUGMENTATION_TRAIN)
    # valid_loader = DataLoader("valid", TRAIN_DIR, False, N_PATCHES, PATCH_SIZE)

    model = EffB4Unetpp_v2((None, None, 1), ch=CH)
    # model.load_weights("/data/volume/sinyu/Exp15/100_0.30665_0.60266.h5")

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
        # class_weight=[],
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
