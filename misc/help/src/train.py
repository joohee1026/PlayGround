import cv2
import math
import pydicom
import numpy as np
from glob import glob
from os.path import join, exists
from os import makedirs
from tensorflow.keras.optimizers import Adam
from loader import DataLoader
from networks import get_model


def train():
    TRAIN_DIR = "/data/train"
    TEST_DIR = "/data/test"
    LOG_DIR = "/data/volume/logs"

    if not exists(LOG_DIR):
        makedirs(LOG_DIR)
        print("Create path : ", LOG_DIR)

    BATCH_SIZE = 1
    N_PATCHES = 4
    PATCH_SIZE = 512
    N_CH = 64

    EPOCHS = 10

    train_loader = DataLoader(TRAIN_DIR, BATCH_SIZE, N_PATCHES, PATCH_SIZE)
    model = get_model((None, None, 1), N_CH, large=True)

    # model.load_weights(join(LOG_DIR, "tmp.h5"))
    model.compile(optimizer=Adam(learning_rate=1e-2), loss='binary_crossentropy', metrics=['binary_crossentropy'])

    model.fit_generator(train_loader, epochs=EPOCHS)
    model.save(join(LOG_DIR, "tmp.h5"), save_format="h5")


if __name__ == "__main__":
    print("Start training ...")
    train()
    print("Done !")
