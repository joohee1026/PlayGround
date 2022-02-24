from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers, Input
from efficientnet.tfkeras import EfficientNetB4
# from cord import CoordinateChannel2D


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


def TransposeConvBlock(x, ch, n, cbam=None):
    if cbam:
        x = ConvBlockAttModule(x)
    t1 = layers.Conv2DTranspose(ch, 3, 2, "same")(x)
    if n == 1: return t1

    if cbam:
        t1 = ConvBlockAttModule(t1)
    t2 = layers.Conv2DTranspose(ch, 3, 2, "same")(t1)
    if n == 2: return t1, t2

    if cbam:
        t2 = ConvBlockAttModule(t2)
    t3 = layers.Conv2DTranspose(ch, 3, 2, "same")(t2)
    if n == 3: return t1, t2, t3

    if cbam:
        t3 = ConvBlockAttModule(t3)
    t4 = layers.Conv2DTranspose(ch, 3, 2, "same")(t3)
    if n == 4: return t1, t2, t3, t4

    if cbam:
        t4 = ConvBlockAttModule(t4)
    t5 = layers.Conv2DTranspose(ch, 3, 2, "same")(t4)
    if n == 5: return t1, t2, t3, t4, t5


def EffB4Unetpp(input_shape=(None, None, 1), ch=16, n_classes=9):
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


def aux_output(x, ch, rbs=1, n_classes=9):
    x = ResidualBlock(x, ch, rbs)
    return layers.Conv2D(n_classes, 1, 1, "same", activation="softmax")(x)


def EffB4Unetpp_aux(input_shape=(None, None, 1), ch=16, rbs=2, n_classes=9):
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
    conv0 = ResidualBlock(conv0, ch*32, rbs)
    conv0_t1, conv0_t2, conv0_t3, conv0_t4, conv0_t5 = TransposeConvBlock(conv0, ch*16, 5)

    conv1 = layers.Concatenate()([conv0_t1, conv]) # 16 16 400
    conv1 = ResidualBlock(conv1, ch*16, rbs)
    conv1_t1, conv1_t2, conv1_t3, conv1_t4 = TransposeConvBlock(conv1, ch*8, 4)
    conv1 = LoadLayer(encoder, n1)

    conv2 = layers.Concatenate()([conv0_t2, conv1_t1, conv1]) # 32 32 352
    conv2 = ResidualBlock(conv2, ch*8, rbs)
    conv2_t1, conv2_t2, conv2_t3 = TransposeConvBlock(conv2, ch*4, 3)
    conv2 = LoadLayer(encoder, n2)

    conv3 = layers.Concatenate()([conv0_t3, conv1_t2, conv2_t1, conv2]) # 64 64 280
    conv3 = ResidualBlock(conv3, ch*4, rbs)
    aux1 = aux_output(conv3, ch*4, 1)
    conv3_t1, conv3_t2 = TransposeConvBlock(conv3, ch*2, 2)
    conv3 = LoadLayer(encoder, n3)

    conv4 = layers.Concatenate()([conv0_t4, conv1_t3, conv2_t2, conv3_t1, conv3]) # 128 128 272
    conv4 = ResidualBlock(conv4, ch*2, rbs)
    aux2 = aux_output(conv4, ch*2, 1)
    conv4_t1 = TransposeConvBlock(conv4, ch, 1)
    conv4 = LoadLayer(encoder, n4)

    conv5 = layers.Concatenate()([conv0_t5, conv1_t4, conv2_t3, conv3_t2, conv4_t1, conv4]) # 256 256 272
    conv5 = ResidualBlock(conv5, ch, rbs)
    aux3 = aux_output(conv5, ch, 1)

    conv6 = layers.Conv2DTranspose(ch, 3, 2, "same")(conv5)
    conv6 = ResidualBlock(conv6, ch, rbs)

    outputs = layers.Conv2D(n_classes, 1, 1, "same", activation="softmax")(conv6)
    return Model(inputs, [aux1, aux2, aux3, outputs])


def EffB4Unetpp_v2(input_shape=(None, None, 1), ch=16, cbam="True", n_classes=9):
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
    conv0_t1, conv0_t2, conv0_t3, conv0_t4, conv0_t5 = TransposeConvBlock(conv0, ch*16, 5, cbam)

    conv1 = layers.Concatenate()([conv0_t1, conv]) # 16 16 400
    conv1 = ResidualBlock(conv1, ch*16, 2)
    conv1_t1, conv1_t2, conv1_t3, conv1_t4 = TransposeConvBlock(conv1, ch*8, 4, cbam)
    conv1 = LoadLayer(encoder, n1)

    conv2 = layers.Concatenate()([conv0_t2, conv1_t1, conv1]) # 32 32 352
    conv2 = ResidualBlock(conv2, ch*8, 2)
    conv2_t1, conv2_t2, conv2_t3 = TransposeConvBlock(conv2, ch*4, 3, cbam)
    conv2 = LoadLayer(encoder, n2)

    conv3 = layers.Concatenate()([conv0_t3, conv1_t2, conv2_t1, conv2]) # 64 64 280
    conv3 = ResidualBlock(conv3, ch*4, 2)
    conv3_t1, conv3_t2 = TransposeConvBlock(conv3, ch*2, 2, cbam)
    conv3 = LoadLayer(encoder, n3)

    conv4 = layers.Concatenate()([conv0_t4, conv1_t3, conv2_t2, conv3_t1, conv3]) # 128 128 272
    conv4 = ResidualBlock(conv4, ch*2, 2)
    conv4_t1 = TransposeConvBlock(conv4, ch, 1, cbam)
    conv4 = LoadLayer(encoder, n4)

    conv5 = layers.Concatenate()([conv0_t5, conv1_t4, conv2_t3, conv3_t2, conv4_t1, conv4]) # 256 256 272
    conv5 = ResidualBlock(conv5, ch, 2)

    conv6 = layers.Conv2DTranspose(ch, 3, 2, "same")(conv5)
    conv6 = ResidualBlock(conv6, ch, 2)

    outputs = layers.Conv2D(n_classes, 1, 1, "same", activation="softmax")(conv6)
    return Model(inputs, outputs)


# def CoEffB4Unetpp(input_shape=(None,None,1), ch=16, cbam=True, n_classes=9):
#     input_shape_c = (input_shape[0], input_shape[1], 3)
#     bm = EffB4Unetpp_v2(input_shape_c, ch, cbam, n_classes)
#
#     inputs = Input(input_shape)
#     cox = CoordinateChannel2D()(inputs)
#
#     output = bm(cox)
#     return Model(inputs, output)
