import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers, Input, activations
from tensorflow.python.keras.layers import Layer, InputSpec
from efficientnet.tfkeras import EfficientNetB4


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


class _CoordinateChannel(Layer):
    def __init__(self, rank,
                 use_radius=False,
                 data_format=None,
                 **kwargs):
        super(_CoordinateChannel, self).__init__(**kwargs)

        self.rank = rank
        self.use_radius = use_radius
        self.data_format = K.image_data_format() if data_format is None else data_format
        self.axis = -1

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[self.axis]

        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={self.axis: input_dim})
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = K.shape(inputs)

        input_shape = [input_shape[i] for i in range(4)]
        batch_shape, dim1, dim2, channels = input_shape

        xx_ones = K.ones(K.stack([batch_shape, dim2]), dtype='int32')
        xx_ones = K.expand_dims(xx_ones, axis=-1)

        xx_range = K.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                          K.stack([batch_shape, 1]))
        xx_range = K.expand_dims(xx_range, axis=1)
        xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
        xx_channels = K.expand_dims(xx_channels, axis=-1)
        xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

        yy_ones = K.ones(K.stack([batch_shape, dim1]), dtype='int32')
        yy_ones = K.expand_dims(yy_ones, axis=1)

        yy_range = K.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                          K.stack([batch_shape, 1]))
        yy_range = K.expand_dims(yy_range, axis=-1)

        yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
        yy_channels = K.expand_dims(yy_channels, axis=-1)
        yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

        xx_channels = K.cast(xx_channels, K.floatx())
        xx_channels = xx_channels / K.cast(dim1 - 1, K.floatx())
        xx_channels = (xx_channels * 2) - 1.

        yy_channels = K.cast(yy_channels, K.floatx())
        yy_channels = yy_channels / K.cast(dim2 - 1, K.floatx())
        yy_channels = (yy_channels * 2) - 1.

        outputs = K.concatenate([inputs, xx_channels, yy_channels], axis=-1)

        if self.use_radius:
            rr = K.sqrt(K.square(xx_channels - 0.5) +
                        K.square(yy_channels - 0.5))
            outputs = K.concatenate([outputs, rr], axis=-1)

        return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[self.axis]

        if self.use_radius and self.rank == 2:
            channel_count = 3
        else:
            channel_count = self.rank

        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] + channel_count
        return tuple(output_shape)

    def get_config(self):
        config = {
            'rank': self.rank,
            'use_radius': self.use_radius,
            'data_format': self.data_format
        }
        base_config = super(_CoordinateChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CoordinateChannel2D(_CoordinateChannel):
    def __init__(self, use_radius=False,
                 data_format=None,
                 **kwargs):
        super(CoordinateChannel2D, self).__init__(
            rank=2,
            use_radius=use_radius,
            data_format=data_format,
            **kwargs
        )

    def get_config(self):
        config = super(CoordinateChannel2D, self).get_config()
        config.pop('rank')
        return config


def print_layer_info(model, suffix=None):
    for i in range(len(model.layers)):
        layer_name = model.layers[i].name
        if suffix is None:
            print(layer_name, "///", model.layers[i].output)
        else:
            if suffix in layer_name:
                print(layer_name, "///", model.layers[i].output)


def LoadLayer(model, name):
    layer_names = [layer.name for layer in model.layers]
    return model.layers[layer_names.index(name)].output


def Residual(inputs, ch=16):
    def Conv(x, filters, k_size, activation=True):
        x = layers.Conv2D(filters, k_size, 1, "same")(x)
        x = layers.BatchNormalization()(x)
        if activation is True:
            x = layers.LeakyReLU()(x)
        return x

    x = layers.LeakyReLU()(inputs)
    x = layers.BatchNormalization()(x)
    blockInput = layers.BatchNormalization()(inputs)
    x = Conv(x, ch, 3)
    x = Conv(x, ch, 3, activation=False)
    return layers.Add()([x, blockInput])


def ResidualBlock(x, ch, n=2):
    x = layers.Conv2D(ch, 3, 1, "same")(x)
    for _ in range(n):
        x = Residual(x, ch)
    return layers.LeakyReLU()(x)


def ConvBlockAttModule(inputs, r=8, k_size=7):
    ch = inputs._shape_tuple()[-1]

    # channel att
    cmx = layers.GlobalMaxPooling2D()(inputs)
    cmx = layers.Reshape((1, 1, ch))(cmx)
    cmx = layers.Dense(ch // r, activation="relu", use_bias=True, kernel_initializer="he_normal",
                       bias_initializer="zeros")(cmx)
    cmx = layers.Dense(ch, use_bias=True, kernel_initializer="he_normal", bias_initializer="zeros")(cmx)

    cax = layers.GlobalAveragePooling2D()(inputs)
    cax = layers.Reshape((1, 1, ch))(cax)
    cax = layers.Dense(ch // r, activation="relu", use_bias=True, kernel_initializer="he_normal",
                       bias_initializer="zeros")(cax)
    cax = layers.Dense(ch, use_bias=True, kernel_initializer="he_normal", bias_initializer="zeros")(cax)

    catt = layers.Activation("sigmoid")(layers.Add()([cax, cmx]))
    catt = layers.Multiply()([inputs, catt])

    # spatial att
    smx = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(catt)
    sax = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(catt)
    satt = layers.Concatenate(axis=3)([sax, smx])
    satt = layers.Conv2D(1, k_size, 1, "same", activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(satt)
    return layers.Multiply()([catt, satt])


def AttentionAugmentation(x, depth_k, depth_v, num_heads):
    if depth_k % num_heads != 0:
        raise ValueError("must depth_k % num_heads == 0")
    if depth_v % num_heads != 0:
        raise ValueError("must depth_v % num_heads == 0")
    if depth_k // num_heads < 1.:
        raise ValueError("must depth_k // num_heads >=1.")
    if depth_v // num_heads < 1.:
        raise ValueError("must depth_v // num_heads >=1.")

    def split_head(inputs):
        tensor_shape = inputs._shape_tuple()
        h, w, c = tensor_shape[1:]
        split = layers.Reshape((h, w,  num_heads, c//num_heads))(inputs)
        transpose_axes = (3, 1, 2, 4)
        return layers.Permute(transpose_axes)(split)

    def combine_heads(inputs):
        transposed = layers.Permute((2, 3, 1, 4))(inputs)
        shape = transposed._shape_tuple()
        a, b, c, d = shape[-4:]
        return layers.Reshape((a, b, c*d))(transposed)

    _, h, w, ch = x._shape_tuple()
    q, k, v = tf.split(x, [depth_k, depth_k, depth_v], axis=-1)

    q = split_head(q)
    k = split_head(k)
    v = split_head(v)

    depth_k_heads = depth_k / num_heads
    q = layers.Lambda(lambda x: x*(depth_k_heads ** -.5))(q)

    qk_shape = (num_heads, h*w, depth_k//num_heads)
    v_shape = (num_heads, h*w, depth_v//num_heads)

    flat_q = layers.Reshape(qk_shape)(q)
    flat_k = layers.Reshape(qk_shape)(k)
    flat_v = layers.Reshape(v_shape)(v)

    logits = layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([flat_q, flat_k])

    weights = K.softmax(logits, axis=-1)
    attn_out = tf.matmul(weights, flat_v)

    attn_out_shape = (num_heads, h, w, depth_v//num_heads)
    attn_out = layers.Reshape(attn_out_shape)(attn_out)
    return combine_heads(attn_out)


def AttAugModule(x, ch=None, k=3, s=1, depth_k=8, depth_v=8, num_heads=4):
    if ch is None:
        ch = x._shape_tuple()[-1]
    # depth_k, depth_v = NormDepth(depth_k, depth_v, ch)
    conv_out = layers.Conv2D(ch-depth_v, k, s, "same", kernel_initializer="he_normal")(x)
    qkv_conv = layers.Conv2D(2*depth_k+depth_v, 1, s)(x)
    attn_out = AttentionAugmentation(qkv_conv, depth_k, depth_v, num_heads)
    attn_out = layers.Conv2D(depth_v, 1)(attn_out)
    return layers.Concatenate(axis=-1)([conv_out, attn_out])


def TransposeConvBlock(x, ch, n, module=None, reduce_dim=None):
    if module == "cbam":
        x = ConvBlockAttModule(x)
    elif module == "aa":
        x = AttAugModule(x)
    if reduce_dim:
        t1 = layers.Conv2DTranspose(ch//2, 3, 2, "same")(x)
    else:
        t1 = layers.Conv2DTranspose(ch, 3, 2, "same")(x)
    if n == 1: return t1

    if module == "cbam":
        t1 = ConvBlockAttModule(t1)
    elif module == "aa":
        x = AttAugModule(x)
    if reduce_dim:
        t2 = layers.Conv2DTranspose(ch//4, 3, 2, "same")(t1)
    else:
        t2 = layers.Conv2DTranspose(ch, 3, 2, "same")(t1)
    if n == 2: return t1, t2

    if module == "cbam":
        t2 = ConvBlockAttModule(t2)
    elif module == "aa":
        x = AttAugModule(x)
    if reduce_dim:
        t3 = layers.Conv2DTranspose(ch//8, 3, 2, "same")(t2)
    else:
        t3 = layers.Conv2DTranspose(ch, 3, 2, "same")(t2)
    if n == 3: return t1, t2, t3

    if module == "cbam":
        t3 = ConvBlockAttModule(t3)
    elif module == "aa":
        x = AttAugModule(x)
    if reduce_dim:
        t4 = layers.Conv2DTranspose(ch//16, 3, 2, "same")(t3)
    else:
        t4 = layers.Conv2DTranspose(ch, 3, 2, "same")(t3)
    if n == 4: return t1, t2, t3, t4

    if module == "cbam":
        t4 = ConvBlockAttModule(t4)
    elif module == "aa":
        x = AttAugModule(x)
    if reduce_dim:
        t5 = layers.Conv2DTranspose(ch//32, 3, 2, "same")(t4)
    else:
        t5 = layers.Conv2DTranspose(ch, 3, 2, "same")(t4)
    if n == 5: return t1, t2, t3, t4, t5


def CUnetConv(x, k=3, ch=16, residual=None):
    out = layers.Conv2D(ch, k, 1, "same")(x)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(ch, k, 1, "same")(out)
    out = layers.BatchNormalization()(out)
    if residual is None:
        return layers.ReLU()(out)
    else:
        return layers.ReLU()(out), out


def CUnet(input_shape=(None, None, 1), k=3, ch=16, n_classes=8):
    inputs = Input(input_shape)
    # contracting path
    x1, skip1 = CUnetConv(inputs, k, ch*4, True)
    x2 = layers.Conv2D(ch*4, k, 2, "same")(x1)
    x2, skip2 = CUnetConv(x2, k, ch*8, True)
    x3 = layers.Conv2D(ch*8, k, 2, "same")(x2)
    x3, skip3 = CUnetConv(x3, k, ch*16, True)
    x4 = layers.Conv2D(ch*16, k, 2, "same")(x3)
    x4, skip4 = CUnetConv(x4, k, ch*32, True)
    x5 = layers.Conv2D(ch*32, k, 2, "same")(x4)
    x5 = CUnetConv(x5, k, ch*64)
    # expanding path
    x6 = layers.Conv2DTranspose(ch*32, k, 2, "same")(x5)
    x6 = layers.BatchNormalization()(x6)
    x6 = layers.Concatenate()([x6, skip4])
    x6 = layers.ReLU()(x6)
    x6 = CUnetConv(x6, k, ch*32)
    x7 = layers.Conv2DTranspose(ch*16, k, 2, "same")(x6)
    x7 = layers.BatchNormalization()(x7)
    x7 = layers.Concatenate()([x7, skip3])
    x7 = layers.ReLU()(x7)
    x7 = CUnetConv(x7, k, ch*16)
    x8 = layers.Conv2DTranspose(ch*8, k, 2, "same")(x7)
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.Concatenate()([x8, skip2])
    x8 = layers.ReLU()(x8)
    x8 = CUnetConv(x8, k, ch*8)
    x9 = layers.Conv2DTranspose(ch*4, k, 2, "same")(x8)
    x9 = layers.BatchNormalization()(x9)
    x9 = layers.Concatenate()([x9, skip1])
    x9 = layers.ReLU()(x9)
    x9 = CUnetConv(x9, k, ch*4)

    outputs = layers.Conv2D(n_classes, 1, 1, activation="sigmoid")(x9)
    return Model(inputs, outputs)



def EffB4Unetpp(input_shape=(None, None, 1), ch=16, n_classes=8):
    encoder = EfficientNetB4(input_shape=input_shape, weights=None, include_top=False)
    inputs = encoder.input

    # print_layer_info(encoder)
    n0 = "block6h_add"
    n1 = "block5f_add"
    n2 = "block3d_add"
    n3 = "block2d_add"
    n4 = "block1b_add"

    conv = LoadLayer(encoder, n0) # 16 16 272
    conv0 = layers.LeakyReLU()(conv)
    conv0 = layers.MaxPool2D(2)(conv0) # 8 8 272
    conv0 = ResidualBlock(conv0, ch*32, 2)
    conv0_t1, conv0_t2, conv0_t3, conv0_t4, conv0_t5 = TransposeConvBlock(conv0, ch*16, 5)

    conv1 = layers.Concatenate()([conv0_t1, conv]) # 16 16 400
    conv1 = ResidualBlock(conv1, ch*16, 2)
    conv1_t1, conv1_t2, conv1_t3, conv1_t4 = TransposeConvBlock(conv1, ch*8, 4)
    conv1 = LoadLayer(encoder, n1)

    conv2 = layers.Concatenate()([conv0_t2, conv1_t1, conv1]) # 32 32 352
    conv2 = ResidualBlock(conv2, ch*8, 2)
    conv2_t1, conv2_t2, conv2_t3 = TransposeConvBlock(conv2, ch*4, 3)
    conv2 = LoadLayer(encoder, n2)

    conv3 = layers.Concatenate()([conv0_t3, conv1_t2, conv2_t1, conv2]) # 64 64 280
    conv3 = ResidualBlock(conv3, ch*4, 2)
    conv3_t1, conv3_t2 = TransposeConvBlock(conv3, ch*2, 2)
    conv3 = LoadLayer(encoder, n3)

    conv4 = layers.Concatenate()([conv0_t4, conv1_t3, conv2_t2, conv3_t1, conv3]) # 128 128 272
    conv4 = ResidualBlock(conv4, ch*2, 2)
    conv4_t1 = TransposeConvBlock(conv4, ch, 1)
    conv4 = LoadLayer(encoder, n4)

    conv5 = layers.Concatenate()([conv0_t5, conv1_t4, conv2_t3, conv3_t2, conv4_t1, conv4]) # 256 256 272
    conv5 = ResidualBlock(conv5, ch, 2)

    conv6 = layers.Conv2DTranspose(ch, 3, 2, "same")(conv5)
    conv6 = ResidualBlock(conv6, ch, 2)

    outputs = layers.Conv2D(n_classes, 1, 1, "same", activation="softmax")(conv6)
    return Model(inputs, outputs)


def EffB4Unetpp_v2(input_shape=(None, None, 1), ch=16, module="cbam", n_classes=8):
    """
    Difference from 'EffB4Unetpp' is that first layer of the decoder, pooling deleted and CBAM module added
    """
    encoder = EfficientNetB4(input_shape=input_shape, weights=None, include_top=None)
    inputs = encoder.input

    # print_layer_info(encoder)
    n0 = "block7b_add" # 16,16,448
    n1 = "block5f_add" # 32,32,160
    n2 = "block3d_add" # 64,64,56
    n3 = "block2d_add" # 128,128,32
    n4 = "block1b_add" # 256,256,24

    conv = LoadLayer(encoder, n0)
    conv0 = layers.LeakyReLU()(conv)
    conv0 = layers.Conv2D(ch*32, 3, 2, "same")(conv0)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.LeakyReLU()(conv0)
    conv0 = ResidualBlock(conv0, ch*32, 2)
    conv0_t1, conv0_t2, conv0_t3, conv0_t4, conv0_t5 = TransposeConvBlock(conv0, ch*16, 5, module)

    conv1 = layers.Concatenate()([conv0_t1, conv]) # 16 16 400
    conv1 = ResidualBlock(conv1, ch*16, 2)
    conv1_t1, conv1_t2, conv1_t3, conv1_t4 = TransposeConvBlock(conv1, ch*8, 4, module)
    conv1 = LoadLayer(encoder, n1)

    conv2 = layers.Concatenate()([conv0_t2, conv1_t1, conv1]) # 32 32 352
    conv2 = ResidualBlock(conv2, ch*8, 2)
    conv2_t1, conv2_t2, conv2_t3 = TransposeConvBlock(conv2, ch*4, 3, module)
    conv2 = LoadLayer(encoder, n2)

    conv3 = layers.Concatenate()([conv0_t3, conv1_t2, conv2_t1, conv2]) # 64 64 280
    conv3 = ResidualBlock(conv3, ch*4, 2)
    conv3_t1, conv3_t2 = TransposeConvBlock(conv3, ch*2, 2, module)
    conv3 = LoadLayer(encoder, n3)

    conv4 = layers.Concatenate()([conv0_t4, conv1_t3, conv2_t2, conv3_t1, conv3]) # 128 128 272
    conv4 = ResidualBlock(conv4, ch*2, 2)
    conv4_t1 = TransposeConvBlock(conv4, ch, 1, module)
    conv4 = LoadLayer(encoder, n4)

    conv5 = layers.Concatenate()([conv0_t5, conv1_t4, conv2_t3, conv3_t2, conv4_t1, conv4]) # 256 256 272
    conv5 = ResidualBlock(conv5, ch, 2)

    conv6 = layers.Conv2DTranspose(ch, 3, 2, "same")(conv5)
    conv6 = ResidualBlock(conv6, ch, 2)

    outputs = layers.Conv2D(n_classes, 1, 1, "same", activation="sigmoid")(conv6)
    return Model(inputs, outputs)


def CoEffB4Unetpp(input_shape=(None,None,1), ch=16, cbam=True, n_classes=9):
    input_shape_c = (input_shape[0], input_shape[1], 3)
    bm = EffB4Unetpp_v2(input_shape_c, ch)

    inputs = Input(input_shape)
    cox = CoordinateChannel2D()(inputs)

    output = bm(cox)
    return Model(inputs, output)


def ASPP(x, ch, atrous_rates=None):
    if atrous_rates is None:
        atrous_rates = (6, 12, 18)

    b4 = layers.GlobalAveragePooling2D()(x)
    b4 = layers.Reshape((1, 1, b4._shape_tuple()[-1]))(b4)
    b4 = layers.Conv2D(ch, 1, 1, "same", use_bias=False)(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.LeakyReLU()(b4)

    size_before = x._shape_tuple()[1:3]
    b4 = layers.Lambda(lambda x: K.resize_images(x, size_before[0], size_before[1], "channels_last", "bilinear"))(b4)

    b0 = layers.Conv2D(ch, 1, 1, "same", use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.LeakyReLU()(b0)

    b1 = SepConvBlock(x, ch, rate=atrous_rates[0], depth_activation=True)
    b2 = SepConvBlock(x, ch, rate=atrous_rates[1], depth_activation=True)
    b3 = SepConvBlock(x, ch, rate=atrous_rates[2], depth_activation=True)

    out = layers.Concatenate()([b4, b0, b1, b2, b3])
    out = layers.Conv2D(ch, 1, 1, "same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    return layers.LeakyReLU()(out)


def EffB4Unetpp_ASPP(input_shape=(None,None,1), ch=16, module="cbam", dout=.1, n_classes=8):
    encoder = EfficientNetB4(input_shape=input_shape, weights=None, include_top=None)
    inputs = encoder.input

    n0 = "block6a_expand_activation" # 32 32 960
    n1 = "block4a_expand_activation" # 64 64 336
    n2 = "block3a_expand_activation" # 128 128 192
    n3 = "block2a_expand_activation" # 256 256 144

    conv = LoadLayer(encoder, n0)
    conv0 = XceptionBlock(conv, [ch*16,ch*16,ch*16], "conv", 1, 1, True)
    conv0 = XceptionBlock(conv0, [ch*16,ch*16,ch*16], "conv", 1, 2, True)
    if dout is not None:
        conv0 = layers.Dropout(dout)(conv0)
    conv0_t1, conv0_t2, conv0_t3 = TransposeConvBlock(conv0, ch*8, 3, module, reduce_dim=False)

    conv1 = LoadLayer(encoder, n1)
    conv1 = layers.Concatenate()([conv0_t1, conv1])
    conv1 = XceptionBlock(conv1, [ch*8,ch*8,ch*8], "conv", 1, 1, True)
    conv1 = XceptionBlock(conv1, [ch*8,ch*8,ch*8], "conv", 1, 2, True)
    conv1 = ASPP(conv1, ch*8)
    if dout is not None:
        conv1 = layers.Dropout(dout)(conv1)
    conv1_t1, conv1_t2 = TransposeConvBlock(conv1, ch*4, 2, module, reduce_dim=False)

    conv2 = LoadLayer(encoder, n2)
    conv2 = layers.Concatenate()([conv0_t2, conv1_t1, conv2])
    conv2 = XceptionBlock(conv2, [ch*4,ch*4,ch*4], "conv", 1, 1, True)
    conv2 = XceptionBlock(conv2, [ch*4,ch*4,ch*4], "conv", 1, 2, True)
    conv2 = ASPP(conv2, ch*8)
    if dout is not None:
        conv2 = layers.Dropout(dout)(conv2)
    conv2_t1 = TransposeConvBlock(conv2, ch*2, 1, module, reduce_dim=False)

    conv3 = LoadLayer(encoder, n3)
    conv3 = layers.Concatenate()([conv0_t3, conv1_t2, conv2_t1, conv3])
    conv3 = XceptionBlock(conv3, [ch*2,ch*2,ch*2], "conv", 1, 1, True)
    conv3 = XceptionBlock(conv3, [ch*2,ch*2,ch*2], "conv", 1, 2, True)
    conv3 = ASPP(conv3, ch*8)
    if dout is not None:
        conv3 = layers.Dropout(dout)(conv3)

    conv3 = TransposeConvBlock(conv3, ch, 1, module, reduce_dim=False)
    # conv3 = XceptionBlock(conv3, [ch,ch,ch], "conv", 1, 2, True)
    # conv3 = ASPP(conv3, ch*8)

    outputs = layers.Conv2D(n_classes, 1, 1, "same", activation="sigmoid")(conv3)
    return Model(inputs, outputs)


def CoEffB4Unetpp_ASPP(input_shape=(None,None,1), ch=16, module="cbam", dout=None, n_classes=8):
    input_ch = input_shape[-1]
    input_shape_c = (input_shape[0], input_shape[1], 3 if input_ch == 1 else 5)
    bm = EffB4Unetpp_ASPP(input_shape_c, ch, module, dout, n_classes)
    # bm.load_weights("/data/volume/sinyu/Exp28/120_0.30884_0.55158.h5")

    inputs = Input(input_shape)
    cox = CoordinateChannel2D()(inputs)

    output = bm(cox)
    return Model(inputs, output)
