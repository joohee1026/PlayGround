import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers, Input
from tensorflow.python.keras.layers import Layer, InputSpec
from efficientnet.tfkeras import EfficientNetB4, EfficientNetB5


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


def TransposeConvBlock(x, ch, n, module=None):
    if module == "cbam":
        x = ConvBlockAttModule(x)
    elif module == "aa":
        x = AttAugModule(x)
    t1 = layers.Conv2DTranspose(ch, 3, 2, "same")(x)
    if n == 1: return t1

    if module == "cbam":
        t1 = ConvBlockAttModule(t1)
    elif module == "aa":
        t1 = AttAugModule(t1)
    t2 = layers.Conv2DTranspose(ch, 3, 2, "same")(t1)
    if n == 2: return t1, t2

    if module == "cbam":
        t2 = ConvBlockAttModule(t2)
    elif module == "aa":
        t2 = AttAugModule(t2)
    t3 = layers.Conv2DTranspose(ch, 3, 2, "same")(t2)
    if n == 3: return t1, t2, t3

    if module == "cbam":
        t3 = ConvBlockAttModule(t3)
    elif module == "aa":
        t3 = AttAugModule(t3)
    t4 = layers.Conv2DTranspose(ch, 3, 2, "same")(t3)
    if n == 4: return t1, t2, t3, t4

    if module == "cbam":
        t4 = ConvBlockAttModule(t4)
    elif module == "aa":
        t4 = AttAugModule(t4)
    t5 = layers.Conv2DTranspose(ch, 3, 2, "same")(t4)
    if n == 5: return t1, t2, t3, t4, t5


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


def CoCrdRegressor(input_shape=(None, None, 1), n_classes=80, mode="b4"):
    """
    EfficientNetB4 Regression
    """
    input_shape_ = (input_shape[0], input_shape[1], 3 if input_shape[-1]==1 else 5)
    if mode == "b4":
        regressor_b = EfficientNetB4(input_shape=input_shape_, weights=None, include_top=None)
    else:
        regressor_b = EfficientNetB5(input_shape=input_shape_, weights=None, include_top=None)
    regressor_gap = layers.GlobalAveragePooling2D()(regressor_b.output)
    regressor_out = layers.Dense(n_classes, activation="relu")(regressor_gap)
    regressor = Model(regressor_b.inputs, regressor_out)

    inputs = Input(input_shape)
    cox = CoordinateChannel2D()(inputs)

    outputs = regressor(cox)
    return Model(inputs, outputs)


def CrdSegPredictor(input_shape=(None, None, 1), ch=8, module=None, n_classes=8):
    """
    Unetpp with EfficientNetB4 Encoder + CBAM module
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

    seg_outputs = layers.Conv2D(n_classes, 1, 1, "same", activation="sigmoid")(conv6)

    crd_outputs = layers.GlobalAveragePooling2D()(encoder.output)
    crd_outputs = layers.Dense(80, activation="relu")(crd_outputs)

    return Model(inputs, [seg_outputs, crd_outputs])


def CoCrdSegPredictor(input_shape=(None,None,1), ch=8, module=None, n_classes=8):
    input_shape_c = (input_shape[0], input_shape[1], 3 if input_shape[-1]==1 else 5)
    bm = CrdSegPredictor(input_shape_c, ch, module, n_classes)

    inputs = Input(input_shape)
    cox = CoordinateChannel2D()(inputs)

    seg_outputs, crd_outputs = bm(cox)
    return Model(inputs, [seg_outputs, crd_outputs])
