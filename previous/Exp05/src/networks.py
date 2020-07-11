from tensorflow.keras import Model, layers, Input
from tensorflow.keras.applications.xception import Xception
from efficientnet.tfkeras import EfficientNetB4


def contract_block(x, out_ch, n=2):
    for _ in range(n):
        x = layers.Conv2D(out_ch, 3, 1, "same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return x


def expand_block(x, out_ch):
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(out_ch, 3, 1, "same")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def contract_block_v2(x, out_ch, first=None):
    x = layers.Conv2D(out_ch, 7, 1, "same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    if first:
        x = layers.Conv2D(out_ch, 7, 1, "same")(x)
    else:
        x = layers.Conv2D(out_ch, 7, 2, "same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    return x


def expand_block_v2(x, out_ch):
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(out_ch, 7, 1, "same")(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU()(x)


def attention_block(g, x, out_ch):
    g = layers.Conv2D(out_ch, 1, 1)(g)
    x = layers.Conv2D(out_ch, 1, 1)(x)
    gx = layers.Add()([g, x])
    gx = layers.ReLU()(gx)
    psi = layers.Conv2D(1, 1, 1, activation='sigmoid')(gx)
    return layers.Multiply()([x, psi])


def attetion_unet(x, out_ch=64, n_classes=8):
    x1 = contract_block(x, out_ch)
    x2 = contract_block(layers.MaxPool2D()(x1), out_ch*2)
    x3 = contract_block(layers.MaxPool2D()(x2), out_ch*4)
    x4 = contract_block(layers.MaxPool2D()(x3), out_ch*8)

    x4 = expand_block(x4, out_ch*4)
    x4_att = attention_block(x4, x3, out_ch*4)
    x4 = layers.Concatenate(axis=-1)([x4, x4_att])

    x3 = expand_block(x4, out_ch*2)
    x3_att = attention_block(x3, x2, out_ch*2)
    x3 = layers.Concatenate(axis=-1)([x3, x3_att])

    x2 = expand_block(x3, out_ch)
    x2_att = attention_block(x2, x1, out_ch)
    x2 = layers.Concatenate(axis=-1)([x2, x2_att])

    return layers.Conv2D(n_classes, 1, 1, activation='softmax')(x2)


def attetion_unet_large(x, out_ch=64, n_classes=8):
    x1 = contract_block(x, out_ch)
    x2 = contract_block(layers.MaxPool2D()(x1), out_ch * 2)
    x3 = contract_block(layers.MaxPool2D()(x2), out_ch * 4)
    x4 = contract_block(layers.MaxPool2D()(x3), out_ch * 8)
    x5 = contract_block(layers.MaxPool2D()(x4), out_ch * 12)
    x6 = contract_block(layers.MaxPool2D()(x5), out_ch * 16)

    x6 = expand_block(x6, out_ch * 12)
    x6_att = attention_block(x6, x5, out_ch * 12)
    x6 = layers.Concatenate(axis=-1)([x6, x6_att])

    x5 = expand_block(x6, out_ch * 8)
    x5_att = attention_block(x5, x4, out_ch * 8)
    x5 = layers.Concatenate(axis=-1)([x5, x5_att])

    x4 = expand_block(x5, out_ch * 4)
    x4_att = attention_block(x4, x3, out_ch * 4)
    x4 = layers.Concatenate(axis=-1)([x4, x4_att])

    x3 = expand_block(x4, out_ch * 2)
    x3_att = attention_block(x3, x2, out_ch * 2)
    x3 = layers.Concatenate(axis=-1)([x3, x3_att])

    x2 = expand_block(x3, out_ch)
    x2_att = attention_block(x2, x1, out_ch)
    x2 = layers.Concatenate(axis=-1)([x2, x2_att])

    return layers.Conv2D(n_classes, 1, 1, activation='sigmoid')(x2)


def attention_unet_large_v2(x, out_ch=64, n_classes=8):
    x1 = contract_block_v2(x, out_ch, first=True)
    x2 = contract_block_v2(x1, out_ch*2)
    x3 = contract_block_v2(x2, out_ch*4)
    x4 = contract_block_v2(x3, out_ch*8)
    x5 = contract_block_v2(x4, out_ch*10)
    # x6 = contract_block_v2(x5, out_ch*12)
    #
    # x6 = expand_block_v2(x6, out_ch*10)
    # x6_att = attention_block(x6, x5, out_ch*10)
    # x6 = layers.Concatenate(axis=-1)([x6, x6_att])

    x5 = expand_block_v2(x5, out_ch*8)
    x5_att = attention_block(x5, x4, out_ch*8)
    x5 = layers.Concatenate(axis=-1)([x5, x5_att])

    x4 = expand_block_v2(x5, out_ch*4)
    x4_att = attention_block(x4, x3, out_ch*4)
    x4 = layers.Concatenate(axis=-1)([x4, x4_att])

    x3 = expand_block_v2(x4, out_ch*2)
    x3_att = attention_block(x3, x2, out_ch*2)
    x3 = layers.Concatenate(axis=-1)([x3, x3_att])

    x2 = expand_block_v2(x3, out_ch)
    x2_att = attention_block(x2, x1, out_ch)
    x2 = layers.Concatenate(axis=-1)([x2, x2_att])

    return layers.Conv2D(n_classes, 1, 1, activation="sigmoid")(x2)


def get_model(input_shape, out_ch=64, n_classes=8, large=False):
    inputs = Input(input_shape)
    if large:
        outputs = attention_unet_large_v2(inputs, out_ch, n_classes)
    else:
        outputs = attetion_unet(inputs, out_ch, n_classes)
    return Model(inputs=[inputs], outputs=[outputs])


def conv_block(x, filters, size, strides=1, padding="same", activation=True):
    x = layers.Conv2D(filters, size, strides, padding)(x)
    x = layers.BatchNormalization()(x)
    if activation is True:
        x = layers.LeakyReLU()(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = layers.LeakyReLU(.1)(blockInput)
    x = layers.BatchNormalization()(x)
    blockInput = layers.BatchNormalization()(blockInput)
    x = conv_block(x, num_filters, 3)
    x = conv_block(x, num_filters, 3, activation=False)
    x = layers.Add()([x, blockInput])
    return x

def custom_xception(input_shape=(None, None, 1), dropout_rate=.3):
    backbone = Xception(input_shape=input_shape, weights=None, include_top=False)
    _input = backbone.input
    ch = 8

    layer_names = [layer.name for layer in backbone.layers]

    conv4 = backbone.layers[layer_names.index("add_11")].output
    conv4 = layers.LeakyReLU()(conv4)
    pool4 = layers.MaxPool2D(2)(conv4)
    pool4 = layers.Dropout(dropout_rate)(pool4)

    convm = layers.Conv2D(ch*32, 3, 1, "same")(pool4)
    convm = residual_block(convm, ch*32)
    convm = residual_block(convm, ch*32)
    convm = layers.LeakyReLU()(convm)

    deconv4 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(convm)
    deconv4_up1 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4)
    deconv4_up2 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4_up1)
    deconv4_up3 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4_up2)
    deconv4_up4 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4_up3)
    uconv4 = layers.Concatenate()([deconv4, conv4])
    uconv4 = layers.Dropout(dropout_rate)(uconv4)

    uconv4 = layers.Conv2D(ch*16, 3, 1, "same")(uconv4)
    uconv4 = residual_block(uconv4, ch*16)
    uconv4 = residual_block(uconv4, ch*16)
    uconv4 = layers.LeakyReLU()(uconv4)

    deconv3 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(uconv4)
    deconv3_up1 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(deconv3)
    deconv3_up2 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(deconv3_up1)
    deconv3_up3 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(deconv3_up2)
    conv3 = backbone.layers[layer_names.index("add_10")].output
    uconv3 = layers.Concatenate()([deconv4_up1, deconv3, conv3])
    uconv3 = layers.Dropout(dropout_rate)(uconv3)

    uconv3 = layers.Conv2D(ch*8, 3, 1, "same")(uconv3)
    uconv3 = residual_block(uconv3, ch*8)
    uconv3 = residual_block(uconv3, ch*8)
    uconv3 = layers.LeakyReLU()(uconv3)

    deconv2 = layers.Conv2DTranspose(ch*4, 3, 2, "same")(uconv3)
    deconv2_up1 = layers.Conv2DTranspose(ch*4, 3, 2, "same")(deconv2)
    deconv2_up2 = layers.Conv2DTranspose(ch*4, 3, 2, "same")(deconv2_up1)
    conv2 = backbone.layers[layer_names.index("add_1")].output
    uconv2 = layers.Concatenate()([deconv4_up2, deconv3_up1, deconv2, conv2])
    uconv2 = layers.Dropout(dropout_rate)(uconv2)

    uconv2 = layers.Conv2D(ch*4, 3, 1, "same")(uconv2)
    uconv2 = residual_block(uconv2, ch*4)
    uconv2 = residual_block(uconv2, ch*4)
    uconv2 = layers.LeakyReLU()(uconv2)

    deconv1 = layers.Conv2DTranspose(ch*2, 3, 2, "same")(uconv2)
    deconv1_up1 = layers.Conv2DTranspose(ch*2, 3, 2, "same")(deconv1)
    conv1 = backbone.layers[layer_names.index("add")].output
    conv1 = layers.ZeroPadding2D(((1,0),(1,0)))(conv1)
    uconv1 = layers.Concatenate()([deconv4_up3, deconv3_up2, deconv2_up1, deconv1, conv1])
    uconv1 = layers.Dropout(dropout_rate)(uconv1)

    uconv1 = layers.Conv2D(ch*2, 3, 1, "same")(uconv1)
    uconv1 = residual_block(uconv1, ch*2)
    uconv1 = residual_block(uconv1, ch*2)
    uconv1 = layers.LeakyReLU()(uconv1)

    uconv0 = layers.Conv2DTranspose(ch*1, 3, 2, "same")(uconv1)
    conv0 = backbone.layers[layer_names.index("block2_sepconv2_bn")].output
    conv0 = layers.ZeroPadding2D(((3,0),(3,0)))(conv0)
    conv0 = layers.LeakyReLU()(conv0)
    uconv0 = layers.Concatenate()([deconv4_up4, deconv3_up3, deconv2_up2, deconv1_up1, uconv0, conv0])

    uconv0 = layers.Dropout(dropout_rate)(uconv0)
    uconv0 = layers.Conv2D(ch*1, 3, 1, "same")(uconv0)
    uconv0 = residual_block(uconv0, ch*1)
    uconv0 = residual_block(uconv0, ch*1)
    uconv0 = layers.LeakyReLU(.1)(uconv0)

    uconv = layers.Conv2DTranspose(ch*1, 3, 2, "same")(uconv0)
    uconv = residual_block(uconv, ch*1)
    output_layer = layers.Conv2D(8, 1, 1, "same", activation="sigmoid")(uconv)

    model = Model(_input, output_layer)
    return model


def custom_B4(input_shape=(None, None, 1), n_classes=8):
    backbone = EfficientNetB4(input_shape=input_shape, weights=None, include_top=False)
    inputs = backbone.input
    ch = 8

    layer_names = [layer.name for layer in backbone.layers]

    conv4 = backbone.layers[layer_names.index("block6h_add")].output # 8, 8, -
    conv4 = layers.LeakyReLU()(conv4)
    pool4 = layers.MaxPool2D(2)(conv4)

    convm = layers.Conv2D(ch*32, 3, 1, "same")(pool4)
    convm = residual_block(convm, ch*32)
    convm = residual_block(convm, ch*32)
    convm = layers.LeakyReLU()(convm)

    deconv4 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(convm)
    deconv4_up1 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4)
    deconv4_up2 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4_up1)
    deconv4_up3 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4_up2)
    deconv4_up4 = layers.Conv2DTranspose(ch*16, 3, 2, "same")(deconv4_up3)
    uconv4 = layers.Concatenate()([deconv4, conv4])

    uconv4 = layers.Conv2D(ch*16, 3, 1, "same")(uconv4)
    uconv4 = residual_block(uconv4, ch*16)
    uconv4 = residual_block(uconv4, ch*16)
    uconv4 = layers.LeakyReLU()(uconv4)

    deconv3 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(uconv4)
    deconv3_up1 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(deconv3)
    deconv3_up2 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(deconv3_up1)
    deconv3_up3 = layers.Conv2DTranspose(ch*8, 3, 2, "same")(deconv3_up2)
    conv3 = backbone.layers[layer_names.index("block5f_add")].output # 16, 16, -
    uconv3 = layers.Concatenate()([deconv3, deconv4_up1, conv3])

    uconv3 = layers.Conv2D(ch*8, 3, 1, "same")(uconv3)
    uconv3 = residual_block(uconv3, ch*8)
    uconv3 = residual_block(uconv3, ch*8)
    uconv3 = layers.LeakyReLU()(uconv3)

    deconv2 = layers.Conv2DTranspose(ch*4, 3, 2, "same")(uconv3)
    deconv2_up1 = layers.Conv2DTranspose(ch*4, 3, 2, "same")(deconv2)
    deconv2_up2 = layers.Conv2DTranspose(ch*4, 3, 2, "same")(deconv2_up1)
    conv2 = backbone.layers[layer_names.index("block3d_add")].output
    uconv2 = layers.Concatenate()([deconv2, deconv3_up1, deconv4_up2, conv2])

    uconv2 = layers.Conv2D(ch*4, 3, 1, "same")(uconv2)
    uconv2 = residual_block(uconv2, ch*4)
    uconv2 = residual_block(uconv2, ch*4)
    uconv2 = layers.LeakyReLU()(uconv2)

    deconv1 = layers.Conv2DTranspose(ch*2, 3, 2, "same")(uconv2)
    deconv1_up1 = layers.Conv2DTranspose(ch*2, 3, 2, "same")(deconv1)
    conv1 = backbone.layers[layer_names.index("block2d_add")].output
    uconv1 = layers.Concatenate()([deconv1, deconv2_up1, deconv3_up2, deconv4_up3, conv1])

    uconv1 = layers.Conv2D(ch*2, 3, 1, "same")(uconv1)
    uconv1 = residual_block(uconv1, ch*2)
    uconv1 = residual_block(uconv1, ch*2)
    uconv1 = layers.LeakyReLU()(uconv1)

    deconv0 = layers.Conv2DTranspose(ch*1, 3, 2, "same")(uconv1)
    conv0 = backbone.layers[layer_names.index("block1b_add")].output
    uconv0 = layers.Concatenate()([deconv4_up4, deconv3_up3, deconv2_up2, deconv1_up1, deconv0, conv0])

    uconv0 = layers.Conv2D(ch*1, 3, 1, "same")(uconv0)
    uconv0 = residual_block(uconv0, ch*1)
    uconv0 = residual_block(uconv0, ch*1)
    uconv0 = layers.LeakyReLU()(uconv0)

    uconv = layers.Conv2DTranspose(ch, 3, 2, "same")(uconv0)
    uconv = residual_block(uconv, ch)
    uconv = residual_block(uconv, ch)
    uconv = layers.LeakyReLU()(uconv)

    outputs = layers.Conv2D(n_classes, 1, 1, "same", activation="sigmoid")(uconv)
    return Model(inputs, outputs)

