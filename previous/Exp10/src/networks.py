from tensorflow.keras import Model, layers, Input, regularizers


def SqueezeExciteBlock(x, ratio=16, spatial=None, channel=None):
    def normal_se(x, ratio=ratio):
        channel_axis = -1
        filters = x._shape_tuple()[channel_axis]
        se_shape = (1, 1, filters)

        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Reshape(se_shape)(se)
        se = layers.Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=False)(se)
        se = layers.Dense(filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(se)
        return layers.multiply([x, se])

    if (spatial is None) and (channel is None):
        return normal_se(x, ratio)

    if (spatial is not None) and (channel is None):
        se = layers.Conv2D(1, (1, 1), activation="sigmoid", use_bias=False, kernel_initializer="he_normal")(x)
        return layers.multiply([x, se])

    if (spatial is not None) and (channel is not None):
        nse = normal_se(x, ratio)
        sse = layers.multiply([x, layers.Conv2D(1, (1, 1), activation="sigmoid", use_bias=False, kernel_initializer="he_normal")(x)])
        return layers.add([nse, sse])


def BasicConv(x, weight_decay=5e-4):
    x = layers.Conv2D(64, 3, 1, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU()(x)


def GroupConv(inputs, grouped_channels, cardinality, strides, weight_decay=5e-4):
    init = inputs
    if cardinality == 1:
        x = layers.Conv2D(grouped_channels, 3, strides, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(init)
        x = layers.BatchNormalization()(x)
        return layers.LeakyReLU()(x)

    group_list = []
    for c in range(cardinality):
        x = layers.Lambda(lambda z: z[:, :, :, c*grouped_channels:(c+1)*grouped_channels])(inputs)
        x = layers.Conv2D(grouped_channels, 3, strides, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
        group_list.append(x)

    group_merge = layers.concatenate(group_list, axis=-1)
    x = layers.BatchNormalization()(group_merge)
    return layers.LeakyReLU()(x)


def ConvInception(x, weight_decay=5e-4):
    x = layers.Conv2D(64, 7, 2, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    return layers.MaxPool2D(3, 2, padding="same")(x)


def BottleneckConv(x, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    inputs = x
    grouped_channels = int(filters / cardinality)
    if x._shape_tuple()[-1] != 2 * filters:
        inputs = layers.Conv2D(filters*2, 1, strides, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        inputs = layers.BatchNormalization()(inputs)

    x = layers.Conv2D(filters, 1, 1, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = GroupConv(x, grouped_channels, cardinality, strides, weight_decay)
    x = layers.Conv2D(filters*2, 1, 1, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = SqueezeExciteBlock(x)
    x = layers.add([inputs, x])
    return layers.LeakyReLU()(x)


def __SEResNeXt(x, depth=29, cardinality=8, width=4, weight_decay=5e-4, include_top=False, pooling=None, n_classes=None):
    if isinstance(depth, list) or isinstance(depth, tuple):
        N = list(depth)
    else:
        N = [(depth - 2) // 9 for _ in range(3)]

    filters = cardinality * width
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2

    x = BasicConv(x, weight_decay)

    for i in range(N[0]):
        x = BottleneckConv(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)

    N = N[1:]
    filters_list = filters_list[1:]

    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = BottleneckConv(x, filters_list[block_idx], cardinality, 2, weight_decay)
            else:
                x = BottleneckConv(x, filters_list[block_idx], cardinality, 1, weight_decay)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(n_classes, use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay), activation="softmax")(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        if pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)
    return x


def _SEResNeXt(x, depth, cardinality=32, width=4, weight_decay=5e-4, include_top=None, pooling=None, n_classes=None):
    if isinstance(depth, list) or isinstance(depth, tuple):
        N = list(depth)
    else:
        N = [(depth - 2) // 9 for _ in range(3)]

    filters = cardinality * width
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2

    x = ConvInception(x, weight_decay)

    for i in range(N[0]):
        x = BottleneckConv(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)

    N = N[1:]
    filters_list = filters_list[1:]

    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = BottleneckConv(x, filters_list[block_idx], cardinality, 2, weight_decay)
            else:
                x = BottleneckConv(x, filters_list[block_idx], cardinality, 1, weight_decay)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(n_classes, use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay), activation="softmax")(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        if pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)
    return x



def SEResNeXt50(input_shape=(None, None, 1), depth=[3,4,6,3], cardinality=32, width=4, weight_decay=5e-4, include_top=False, pooling=None, n_classes=None):
    inputs = Input(shape=input_shape)
    x = _SEResNeXt(inputs, depth, cardinality, width, weight_decay, include_top, pooling, n_classes)
    return Model(inputs, x)


def SEResNeXt101(input_shape=(None, None, 1), depth=[3,4,23,3], cardinality=32, width=4, weight_decay=5e-4, include_top=False, pooling=None, n_classes=None):
    inputs = Input(shape=input_shape)
    x = _SEResNeXt(inputs, depth, cardinality, width, weight_decay, include_top, pooling, n_classes)
    return Model(inputs, x)


def ResidualBlock(blockInput, num_filters=16):
    def Conv(x, filters, size, strides=1, padding="same", activation=True):
        x = layers.Conv2D(filters, size, strides, padding)(x)
        x = layers.BatchNormalization()(x)
        if activation is True:
            x = layers.LeakyReLU()(x)
        return x

    x = layers.LeakyReLU()(blockInput)
    x = layers.BatchNormalization()(x)
    blockInput = layers.BatchNormalization()(blockInput)
    x = Conv(x, num_filters, 3)
    x = Conv(x, num_filters, 3, activation=False)
    x = layers.Add()([x, blockInput])
    return x


def SEResNeXtUnetpp(input_shape=(None, None, 1), ch=16, n_classes=9):
    backbone = SEResNeXt50(input_shape)
    inputs = backbone.input

    layer_names = [layer.name for layer in backbone.layers]
    # for name in layer_names:
    #     print(name, "//", backbone.layers[layer_names.index(name)].output)

    layer_4 = "leaky_re_lu_48" # last output
    layer_3 = "leaky_re_lu_38" # 32 32 512
    layer_2 = "leaky_re_lu_20" # 64 64 256
    layer_1 = "leaky_re_lu_8" # 128 128 128

    conv_last = backbone.layers[layer_names.index(layer_4)].output
    conv5 = layers.Conv2D(ch*32, 3, 2, "same")(conv_last)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.LeakyReLU()(conv5)

    deconv4 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(conv5)
    deconv4_up1 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4)
    deconv4_up2 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4_up1)
    deconv4_up3 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4_up2)
    deconv4_up4 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4_up3)

    conv4 = layers.Concatenate()([deconv4, conv_last])
    conv4 = layers.Conv2D(ch*16, 3, 1, "same")(conv4)
    conv4 = ResidualBlock(conv4, ch*16)
    conv4 = ResidualBlock(conv4, ch*16)
    conv4 = layers.LeakyReLU()(conv4)

    deconv3 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(conv4)
    deconv3_up1 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(deconv3)
    deconv3_up2 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(deconv3_up1)
    deconv3_up3 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(deconv3_up2)

    conv3 = backbone.layers[layer_names.index(layer_3)].output
    conv3 = layers.Concatenate()([deconv3, deconv4_up1, conv3])
    conv3 = layers.Conv2D(ch*8, 3, 1, "same")(conv3)
    conv3 = ResidualBlock(conv3, ch*8)
    conv3 = ResidualBlock(conv3, ch*8)
    conv3 = layers.LeakyReLU()(conv3)

    deconv2 = layers.Conv2DTranspose(ch*4, 3, 2, "same")(conv3)
    deconv2_up1 = layers.Conv2DTranspose(ch*4, 3, 2, "same")(deconv2)
    deconv2_up2 = layers.Conv2DTranspose(ch*4, 3, 2, "same")(deconv2_up1)

    conv2 = backbone.layers[layer_names.index(layer_2)].output
    conv2 = layers.Concatenate()([deconv2, deconv4_up2,deconv3_up1, conv2])
    conv2 = layers.Conv2D(ch*4, 3, 1, "same")(conv2)
    conv2 = ResidualBlock(conv2, ch*4)
    conv2 = ResidualBlock(conv2, ch*4)
    conv2 = layers.LeakyReLU()(conv2)

    deconv1 = layers.Conv2DTranspose(ch*2, 3, 2, "same")(conv2)
    deconv1_up1 = layers.Conv2DTranspose(ch*2, 3, 2, "same")(deconv1)

    conv1 = backbone.layers[layer_names.index(layer_1)].output
    conv1 = layers.Concatenate()([deconv1, deconv4_up3, deconv3_up2, deconv2_up1, conv1])
    conv1 = layers.Conv2D(ch*2, 3, 1, "same")(conv1)
    conv1 = ResidualBlock(conv1, ch*2)
    conv1 = ResidualBlock(conv1, ch*2)
    conv1 = layers.LeakyReLU()(conv1)

    deconv0 = layers.Conv2DTranspose(ch, 3, 2, "same")(conv1)

    conv0 = layers.Concatenate()([deconv0, deconv4_up4, deconv3_up3, deconv2_up2, deconv1_up1])
    conv0 = layers.Conv2D(ch, 3, 1, "same")(conv0)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.LeakyReLU()(conv0)
    conv0 = ResidualBlock(conv0, ch)

    deconv = layers.Conv2DTranspose(ch, 7, 2, "same")(conv0)
    deconv = layers.BatchNormalization()(deconv)
    deconv = layers.LeakyReLU()(deconv)

    deconv = layers.Conv2D(ch, 3, 1, "same")(deconv)
    deconv = ResidualBlock(deconv, ch)
    deconv = ResidualBlock(deconv, ch)
    deconv = layers.LeakyReLU()(deconv)

    outputs = layers.Conv2D(n_classes, 1, 1, "same", activation="softmax")(deconv)
    return Model(inputs, outputs)
