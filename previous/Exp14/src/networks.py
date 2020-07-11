import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers, Input, activations


def ConvBlock(x, ch, k=3, s=1, rate=1):
    if s == 1:
        return layers.Conv2D(ch, k, s, "same", use_bias=False, dilation_rate=rate)(x)
    else:
        k_effective = k + (k-1) * (rate-1)
        pad_beg = (k_effective-1) // 2
        pad_end = (k_effective-1) - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        return layers.Conv2D(ch, k, s, "valid", use_bias=False, dilation_rate=rate)(x)


def SepConvBlock(x, ch, k=3, s=1, rate=1, depth_activation=False):
    if s == 1:
        depth_padding = "same"
    else:
        depth_padding = "valid"
        k_effective = k + (k -1) * (rate - 1)
        pad_beg = (k_effective-1) // 2
        pad_end = (k_effective-1) - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)

    if not depth_activation:
        x = layers.Activation("relu")(x)

    x = layers.DepthwiseConv2D(k, s, depth_padding, dilation_rate=rate, use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if depth_activation:
        x = layers.Activation("relu")(x)

    x = layers.Conv2D(ch, 1, 1, "same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if depth_activation:
        x = layers.Activation("relu")(x)
    return x


def XceptionBlock(x, depth_list, skip_connection_type, stride, rate=1, depth_activation=False, return_skip=False):
    # assert skip_connection_type in ["conv", "sum", "none"]
    residual = x
    for i in range(3):
        residual = SepConvBlock(residual, depth_list[i], s=stride if i ==2 else 1, rate=rate, depth_activation=depth_activation)
        if i == 1:
            skip = residual

    if skip_connection_type == "conv":
        shortcut = ConvBlock(x, depth_list[-1], k=1, s=stride)
        shortcut = layers.BatchNormalization()(shortcut)
        out = layers.Add()([residual, shortcut])
    elif skip_connection_type == "sum":
        out = layers.Add()([residual, x])
    elif skip_connection_type == "none":
        out = residual

    if return_skip:
        return out, skip
    else:
        return out



def DeepLabV3p_Xc(input_shape=(None, None, 1), output_stride=8, n_classes=9):
    if output_stride == 8:
        entry_block3_stride = 1
        middle_block_rate = 2
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    # backbone
    inputs = Input(shape=input_shape)
    out = layers.Conv2D(32, 3, 2, "same", use_bias=False)(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)

    out = ConvBlock(out, 64, 3, 1)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)

    out = XceptionBlock(out, [128,128,128], "conv", 2, 1, False)
    out, skip1 = XceptionBlock(out, [256,256,256], "conv", 2, 1, False, True)
    out = XceptionBlock(out, [728,728,728], "conv", entry_block3_stride, 1, False)
    for _ in range(16):
        out = XceptionBlock(out, [728,728,728], "sum", 1, middle_block_rate, False)
    out = XceptionBlock(out, [728,1024,1024], "conv", 1, exit_block_rates[0], depth_activation=False)
    out = XceptionBlock(out, [1536,1536,2048], "none", 1, exit_block_rates[1], depth_activation=True)

    # image feature branch
    b4 = layers.GlobalAveragePooling2D()(out)
    b4 = layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = layers.Conv2D(256, 1, 1, "same", use_bias=False)(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.ReLU()(b4)

    # size_before = K.int_shape(out)
    size_before = out._shape_tuple()
    b4 = layers.Lambda(lambda x: tf.compat.v1.image.resize(x, (tf.constant(size_before[1]), tf.constant(size_before[2])), method='bilinear', align_corners=True))(b4)
    # b4 = layers.Lambda(lambda x: tf.image.resize(x, (size_before[1], size_before[2])))(b4)

    b0 = layers.Conv2D(256, 1, 1, "same", use_bias=False)(out)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.ReLU()(b0)

    b1 = SepConvBlock(out, 256, rate=atrous_rates[0], depth_activation=True)
    b2 = SepConvBlock(out, 256, rate=atrous_rates[1], depth_activation=True)
    b3 = SepConvBlock(out, 256, rate=atrous_rates[2], depth_activation=True)

    out = layers.Concatenate()([b4, b0, b1, b2, b3])
    out = layers.Conv2D(256, 1, 1, "same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Dropout(.1)(out)

    # decoder
    # skip_shape = K.int_shape(skip1)
    skip_shape = skip1._shape_tuple()
    out = layers.Lambda(lambda x: tf.compat.v1.image.resize(x, (tf.constant(skip_shape[1]), tf.constant(skip_shape[2])), method='bilinear', align_corners=True))(out)
    # out = layers.Lambda(lambda x: tf.image.resize(x, (skip_shape[1], skip_shape[2])))(out)

    decoder_skip1 = layers.Conv2D(48, 1, 1, "same", use_bias=False)(skip1)
    decoder_skip1 = layers.BatchNormalization()(decoder_skip1)
    decoder_skip1 = layers.ReLU()(decoder_skip1)

    out = layers.Concatenate()([out, decoder_skip1])
    out = SepConvBlock(out, 256, depth_activation=True)
    out = SepConvBlock(out, 256, depth_activation=True)

    out = layers.Conv2D(n_classes, 1, 1, "same")(out)
    # size_before_ = K.int_shape(inputs)
    size_before_ = inputs._shape_tuple()
    out = layers.Lambda(lambda x: tf.compat.v1.image.resize(x, (tf.constant(size_before_[1]), tf.constant(size_before_[2])), method='bilinear', align_corners=True))(out)
    # out = layers.Lambda(lambda x: tf.image.resize(x, (size_before_[1], size_before_[2])))(out)
    out = layers.Activation("softmax")(out)
    return Model(inputs, out)
