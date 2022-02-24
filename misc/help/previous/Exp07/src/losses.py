import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import losses, callbacks

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true, pred = np.sum(t), np.sum(p)

        if true == 0:
            metric += (pred == 0)
            continue

        intersection = np.sum(t*p)
        union = true + pred - intersection
        iou = intersection / union

        iou = np.floor(max(0, (iou - .45) * 20)) / 10
        metric += iou

    metric /= batch_size
    return metric


def iou_metric(label, pred):
    return tf.compat.v1.py_func(get_iou_vector, [label, pred>.5], tf.float32)


def _gather_channels(x, indexes):
    if K.image_data_format() == 'channels_last':
        x = K.permute_dimensions(x, (3, 0, 1, 2))
        x = K.gather(x, indexes)
        x = K.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = K.permute_dimensions(x, (1, 0, 2, 3))
        x = K.gather(x, indexes)
        x = K.permute_dimensions(x, (1, 0, 2, 3))
    return x


def get_reduce_axes(per_image):
    axes = [1, 2] if K.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(*xs, indexes=None):
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes) for x in xs]
    return xs


def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x


def average(x, per_image=False, class_weights=None):
    if per_image:
        x = K.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)


def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=1e-5, per_image=False, threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    intersection = K.sum(gt * pr, axis=axes)
    union = K.sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, per_image, class_weights)
    return score


def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=1e-5, per_image=False, threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights)
    return score


def precision(gt, pr, class_weights=1, class_indexes=None, smooth=1e-5, per_image=False, threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp

    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, per_image, class_weights)
    return score


def recall(gt, pr, class_weights=1, class_indexes=None, smooth=1e-5, per_image=False, threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    tp = K.sum(gt * pr, axis=axes)
    fn = K.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights)
    return score


def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    axis = 3 if K.image_data_format() == 'channels_last' else 1
    pr /= K.sum(pr, axis=axis, keepdims=True)

    pr = K.clip(pr, K.epsilon(), 1 - K.epsilon())

    output = gt * K.log(pr) * class_weights
    return - K.mean(output)


def categorical_focal_loss(gt, pr, gamma=0.25, alpha=0.45, class_indexes=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = tf.clip_by_value(pr, 1e-7, 1.0-1e-7)
    # pr = K.clip(pr, K.epsilon(), 1.0 - K.epsilon())
    loss = - gt * (alpha * K.pow((1 - pr), gamma) * K.log(pr))
    return K.mean(loss)


class JaccardLoss(losses.Loss):
    def __init__(self, class_weights=None, class_indexes=None, per_image=False, smooth=1e-5):
        super(JaccardLoss, self).__init__(name="jaccard_loss")
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return 1 - iou_score(y_true, y_pred, class_weights=self.class_weights, class_indexes=self.class_indexes, smooth=self.smooth, per_image=self.per_image )


class DiceLoss(losses.Loss):
    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=1e-5):
        super(DiceLoss, self).__init__(name="dice_loss")
        self.beta = beta
        self.class_weights = class_weights
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return 1 - f_score(y_true, y_pred, beta=self.beta, class_weights=self.class_weights, class_indexes=self.class_indexes, per_image=self.per_image, smooth=self.smooth)


class CatCELoss(losses.Loss):
    def __init__(self, class_weights=1., class_indexes=None):
        super(CatCELoss, self).__init__(name="categorical_crossentropy")
        self.class_weights = class_weights
        self.class_indexes = class_indexes

    def __call__(self, y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred, class_weights=self.class_weights, class_indexes=self.class_indexes)


class CatFocalLoss(losses.Loss):
    def __init__(self, alpha=.25, gamma=.25, class_indexes=None):
        super(CatFocalLoss, self).__init__(name="focal_loss")
        self.alpha = alpha
        self.gamma = gamma
        self.class_indexes = class_indexes

    def __call__(self, y_true, y_pred):
        return categorical_focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma, class_indexes=self.class_indexes)


def iou_(y_true, y_pred):
    return iou_score(y_true, y_pred)

def dice_(y_true, y_pred):
    return (1-f_score(y_true, y_pred))

def focal_(y_true, y_pred):
    return categorical_focal_loss(y_true, y_pred)

def cce_dice_loss(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred) + (1 - f_score(y_true, y_pred))

def cce_dice_focal_loss(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred) + (1 - f_score(y_true, y_pred)) + categorical_focal_loss(y_true, y_pred)
