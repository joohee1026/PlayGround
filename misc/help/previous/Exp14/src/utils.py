import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import losses, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class SWA(callbacks.Callback):
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params["epochs"]
        print("Stochastic weight averaging selected for last {} epochs".format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
        elif epoch > self.swa_epoch:
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * (epoch - self.swa_epoch) + self.model.get_weights()[i]) / ((epoch - self.swa_epoch)+1)
        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print("Final model parameters set to stochastic weight average.")
        self.model.save_weights(self.filepath)
        print("Final stochastic averaged weights saved to file.")


class CustomCallbacks:
    def __init__(self, log_dir, nb_epochs, nb_snapshots, init_lr=.1):
        self.log_dir = log_dir
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, mode_prefix="Model"):
        callback_list = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.log_dir, "{epoch:03d}_{val_loss:.5f}_{val_iou_score:.5f}.h5"),
                monitor="val_loss",
                mode="min",
                save_best_only=False,
                save_weights_only=True,
                verbose=1,
            ),
            callbacks.LearningRateScheduler(schedule=self.cosine_annealing)
        ]
        return callback_list

    def cosine_annealing(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2* cos_out)


def get_patches(img, patch_size=256, pad=16):
    uni_pad = patch_size + pad
    bi_pad = patch_size + pad + pad

    shape = img.shape
    pad_width = ((pad,0), (pad,0), (0,0))
    img = np.pad(img, pad_width, mode="constant")

    patches = []
    patches_core = []
    b = [0,0]
    for x in range(pad, shape[0], patch_size):
        b[0] += 1
        for y in range(pad, shape[1], patch_size):
            b[1] += 1
            patch = img[x-pad:x+uni_pad, y-pad:y+uni_pad, :]

            if patch.shape == (bi_pad,bi_pad,3):
                patches.append(patch)
                patches_core.append([
                    (pad,pad),
                    (uni_pad,uni_pad)
                ])
            else:
                need_p = ((0,bi_pad-patch.shape[0]), (0,bi_pad-patch.shape[1]), (0,0))
                patch_pad = np.pad(patch, need_p, mode="constant")
                patches.append(patch_pad)
                patches_core.append([
                    (pad,pad),
                    (patch.shape[1]-pad, patch.shape[0]-pad)
                ])
    b = [b[0], b[1]//b[0]]
    return patches, patches_core, shape, b


def merge_patches(patches, patches_core, shape, b):
    pad = patches_core[0][0][0]
    patch_size = patches[0].shape[0] - (pad * 2)
    uni_pad = patch_size + pad

    h, w, _ = shape

    results = np.zeros((h, w, 8))
    xc, yc = 0, 0
    for idx, patch in enumerate(patches):
        if (xc//patch_size) == (b[1]-1):
            remain_size = results[yc:(yc+patch_size), xc:(xc+patch_size), :].shape
            results[yc:(yc+patch_size), xc:(xc+patch_size), :] = patch[pad:remain_size[0]+pad, pad:remain_size[1]+pad, :]
            xc = 0
            yc += patch_size

        elif ((xc//patch_size) < (b[1]-1)) and ((yc//patch_size) < (b[0]-1)):
            results[yc:(yc+patch_size), xc:(xc+patch_size), :] = patch[pad:uni_pad, pad:uni_pad, :]
            xc += patch_size

        else:
            remain_size = results[yc:, xc:(xc+patch_size), :].shape
            results[yc:, xc:(xc+patch_size), :] = patch[pad:remain_size[0]+pad, pad:remain_size[1]+pad]
            xc += patch_size
    return results
