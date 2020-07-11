import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import losses, callbacks

#
# def bce_dice_focal(y_true, y_pred):
#     return losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred) + binary_focal_loss(y_true, y_pred)
#
#
# def binary_focal_loss(y_true, y_pred):
#     alpha, gamma = .25, .75
#
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#
#     epsilon = K.epsilon()
#     pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
#     pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
#
#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
#            - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def weighted_ce(y_true, y_pred):
    return  tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, .5) * 2


def bce_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def cce_dice_loss(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred) + (1 - f_score(y_true, y_pred))


def dice_loss(y_true, y_pred):
    return (1 - f_score(y_true, y_pred))


def bce_logdice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def bce_dice_focal_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred) + focal_(y_true, y_pred)


def cce_dice_focal_loss(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred) + (1 - f_score(y_true, y_pred)) + categorical_focal_loss(y_true, y_pred)


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


def categorical_focal_loss(gt, pr, gamma=0.25, alpha=0.75, class_indexes=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    # pr = K.clip(pr, K.epsilon(), 1.0 - K.epsilon())
    loss = - gt * (alpha * K.pow((1 - pr), gamma) * K.log(pr))
    return K.mean(loss)


def iou_(y_true, y_pred):
    return iou_score(y_true, y_pred)


def dice_(y_true, y_pred):
    return (1-f_score(y_true, y_pred))


def focal_(y_true, y_pred):
    return categorical_focal_loss(y_true, y_pred)
