from tensorflow.keras import Model, layers, Input


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

    return layers.Conv2D(n_classes, 1, 1, activation='softmax')(x2)


def get_model(input_shape, out_ch=64, n_classes=8, large=False):
    inputs = Input(input_shape)
    if large:
        outputs = attetion_unet_large(inputs, out_ch, n_classes)
    else:
        outputs = attetion_unet(inputs, out_ch, n_classes)
    return Model(inputs=[inputs], outputs=[outputs])
