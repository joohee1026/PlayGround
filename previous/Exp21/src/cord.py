from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec


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

