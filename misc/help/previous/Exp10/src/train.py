import albumentations as albu
from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam

from loader import PatchDataLoader_randdev
from networks import SEResNeXtUnetpp
from utils import CustomCallbacks
from losses import cce_dice_loss, dice_loss, iou_score


def train():
    ### hyper-parameters ================
    BATCH_SIZE = 8
    IMAGE_SIZE = 512
    VER = "Exp10"
    LR = 1e-3

    EPOCHS = 50
    AUGMENTATION_TRAIN = albu.Compose([
        albu.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
        albu.RandomGamma(gamma_limit=(100,400)),
        albu.GaussNoise(var_limit=(.001, .002)),
        albu.GaussianBlur(),
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.ElasticTransform(alpha=1, sigma=2, approximate=True, p=.1),
        albu.GridDistortion(num_steps=1, p=.1)
    ], p=1)


    ### =================================

    TRAIN_DIR = "/data/train"
    LOG_DIR = f"/data/volume/sinyu/{VER}"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    train_loader = PatchDataLoader_randdev("train", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE, AUGMENTATION_TRAIN)
    valid_loader = PatchDataLoader_randdev("valid", TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE)

    model = SEResNeXtUnetpp((IMAGE_SIZE, IMAGE_SIZE, 1), ch=16)

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
        max_queue_size=30,
        use_multiprocessing=True,
        callbacks=CB.get_callbacks()
    )
    model.save(join(LOG_DIR, f"{VER}_f.h5"), save_format="h5")


if __name__ == "__main__":
    print("Start training ...")
    train()
    print("Done !")
