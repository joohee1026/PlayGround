from tensorflow.keras import Model, layers, Input
from tensorflow.keras.applications.xception import Xception


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
