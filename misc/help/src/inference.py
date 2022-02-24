import cv2
import pydicom
import numpy as np
from glob import glob
from os import makedirs
from os.path import join, exists
import tensorflow as tf


def mm_norm(x):
    return (x - x.min()) / (x.max() - x.min())


def get_grid_patch(x, patch_shape, grid, border):
    if isinstance(patch_shape, int):
        patch_shape = (patch_shape, patch_shape)

    if isinstance(grid, int):
        grid = (grid, grid)

    if isinstance(border, int):
        border = (border, border)

    edge = x.shape
    patches = []

    grid_patch_shape = np.array(patch_shape) - np.array(border)
    grid = np.max(x.shape) // np.array(grid_patch_shape) * 1.5

    x_spacing = np.linspace(0, edge[0] - patch_shape[0], grid[0], endpoint=True, dtype=np.int32)
    y_spacing = np.linspace(0, edge[1] - patch_shape[1], grid[1], endpoint=True, dtype=np.int32)

    # x_width = -(edge[0] - x_spacing[-1] - patch_shape[0]) // 2
    # y_width = -(edge[1] - y_spacing[-1] - patch_shape[1]) // 2
    #
    # pad_width = ((x_width, x_width), (y_width, y_width))
    # x = np.pad(x, pad_width, mode="constant")

    for i in x_spacing:
        for j in y_spacing:
            patches.append(
                x[
                i:i+patch_shape[0],
                j:j+patch_shape[1]
                ]
            )
    spacing = (x_spacing, y_spacing)
    return patches, spacing


def assemble_softmax(patches, spacing, orig_shape, n_classes, border):
    if isinstance(border, int):
        border = (border, border)

    initial_array = np.zeros([orig_shape[0], orig_shape[1], n_classes])
    x_spacing, y_spacing = spacing
    x_spacing[1:] += 1
    y_spacing[1:] += 1

    idx = 0

    for x in x_spacing:
        x = int(x)
        for y in y_spacing:
            y = int(y)
            patch = patches[idx]
            idx += 1

            # patch = patch[
            #     border[0]: -border[0],
            #     border[1]: -border[1]
            # ]

            if x == x_spacing[-1]:
                patch = patch[:-1,:,:]
            if y == y_spacing[-1]:
                patch = patch[:,:-1,:]

            initial_array[
                x:(x+patch.shape[0]),
                y:(y+patch.shape[1])
            ] += patch
    return initial_array


def inference():
    TRAIN_DIR = "/data/train"
    TEST_DIR = "/data/test"
    LOG_DIR = "/data/volume/logs"
    OUTPUT_DIR = "/data/output"

    W = "/data/volume/logs/tmp2.h5"

    model = tf.keras.models.load_model(W)

    test_files = glob(join(TEST_DIR, "*"))

    for i, f in enumerate(test_files):
        CASE_ID = f.split("/")[-1][:-4]
        CASE_DIR = join(OUTPUT_DIR, CASE_ID)

        if not exists(CASE_DIR):
            makedirs(CASE_DIR)

        _f = mm_norm(pydicom.dcmread(f).pixel_array).astype(np.float32)
        orig_shape = _f.shape
        patches, spacing = get_grid_patch(_f, 1024, 3, 16)
        seg_patches = []
        for patch in patches:
            pred = model.predict(patch[np.newaxis,...,np.newaxis])
            seg_patches.append(np.squeeze(pred))

        pred_assemble = assemble_softmax(seg_patches, spacing, orig_shape, 8, 16)
        pred_assemble[pred_assemble >=.5] = 255.
        pred_assemble[pred_assemble < .5] = 0.

        pred_assemble = pred_assemble.astype(np.uint8)

        for i, class_name in enumerate([
            "Aortic Knob",
            "Carina",
            "DAO",
            "LAA",
            "Lt Lower C8",
            "Pulmonary Conus",
            "Rt Lower CB",
            "Rt Upper CB"
        ]):
            cv2.imwrite(join(CASE_DIR, CASE_ID+"_"+class_name+".png"), pred_assemble[...,i])

        print(i, "/", len(test_files))

if __name__ == "__main__":
    inference()
