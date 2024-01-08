from packaging import version

import tensorflow as tf

from common.builder import Builder
import common.models as models

if version.parse(tf.__version__) < version.parse('2.11.0'):
    import tensorflow_addons as tfa
    GroupNorm = tfa.layers.GroupNormalization
else:
    GroupNorm = tf.keras.layers.GroupNormalization

def _linear_attention(query, key, value, scale):
    query = tf.nn.softmax(query, axis=-2)
    key = tf.nn.softmax(key, axis=-1)

    query = query * scale
    context = tf.matmul(key, value, transpose_b=True)
    out = tf.matmul(context, query, transpose_a=True)
    return out

def _attention(query, key, value, scale):
    query = query * scale

    sim = tf.matmul(query, key, transpose_a=True)
    sim = sim - tf.math.reduce_max(sim, axis=-1, keepdims=True)
    attn = tf.nn.softmax(sim, axis=-1)
    out = tf.matmul(attn, value, transpose_b=True)
    return out

def attention_init_builder():
    """ create Builder for attention or linear_attention init """
    return Builder(['num_heads', 'attention_dim', 'strategy'],
            default_values={'num_heads': 4, 'attention_dim': 128,
                'strategy': 'attention'})

def _conv_next_block(dim_in, dim_out, mult=2, activation='gelu'):
    # https://arxiv.org/abs/2201.03545
    s = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim_in, 7, padding='same', groups=dim_in),
            GroupNorm(1),
            tf.keras.layers.Conv2D(dim_out * mult, 3, padding='same', activation=activation),
            GroupNorm(1),
            tf.keras.layers.Conv2D(dim_out, 3, padding='same'),
        ])
    return models.Multiplex([
        s,
        tf.keras.layers.Conv2D(dim_out, 1, padding='valid') if dim_in != dim_out else None,
    ])

class ConvAttention(tf.keras.layers.Layer):
    """
    ConvAttention for images
    """
    def __init__(self, model_dim, channel_dim, params):
        super().__init__()
        self.model_dim = model_dim
        self.channel_dim = channel_dim
        self.num_heads = params['num_heads']
        self.params = params

        attention_dim = params['attention_dim']
        assert attention_dim % self.num_heads == 0
        dim_head = attention_dim / self.num_heads
        self.scale = dim_head**-0.5

        self.qkv = tf.keras.layers.Conv2D(attention_dim * 3, 1, use_bias=False)
        self.out = tf.keras.Sequential([
            tf.keras.layers.Conv2D(model_dim, 1),
            GroupNorm(1),
        ])

    # @tf.function(input_signature=[
        # tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    # ])
    def call(self, inputs):
        """
        Model call implementation
        """
        inputs = tf.ensure_shape(inputs, [None, None, None, self.channel_dim])
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        height = inputs_shape[1]
        width = inputs_shape[2]
        channels = int(self.params['attention_dim'] / self.num_heads)
        qkv = tf.split(self.qkv(inputs), 3, -1)
        query, key, value = tuple([
            tf.reshape(tf.transpose(tens, [0, 3, 1, 2]),
                (batch_size, self.num_heads, channels, height * width))
            for tens in qkv])

        if self.params['strategy'] == 'attention':
            out = _attention(query, key, value, self.scale)
        elif self.params['strategy'] == 'linear_attention':
            out = _linear_attention(query, key, value, self.scale)
        else:
            raise NotImplementedError

        out = tf.transpose(tf.reshape(out,
            (batch_size, -1, height, width)), [0, 2, 3, 1])
        return self.out(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'channel_dim': self.channel_dim,
            'params': self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MinUnetDown(tf.keras.Model):
    """
    MinUnetDown preps image into latent image through a series of convolutions and down samples.
    only does double convolution and downsample. no attentions
    """
    def __init__(self, in_channels, params):
        super().__init__()

        self.in_channels = in_channels
        self.params = params

        model_dim = params['model_dim']
        unet_dim = params['unet_dim']
        dim_mults = params['dim_mults']
        convnext_mult = params['convnext_mult']

        init_dim = params.get('init_dim', None)
        if init_dim is None:
            init_dim = unet_dim // 3 * 2

        dims = [init_dim] + [unet_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        self.init_conv = tf.keras.layers.Conv2D(init_dim, 7, padding='same')
        self.downs = [
            _conv_next_block(dim_in, dim_out, mult=convnext_mult)
            for dim_in, dim_out in in_out
        ]
        self.downsamples = [
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
            for _ in range(len(in_out)-1)
        ]
        self.attn = _conv_next_block(mid_dim, model_dim, mult=convnext_mult)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ])
    def call(self, inputs):
        """
        Model call implementation
        """

        img_dim = self.params['image_dim']
        inputs = tf.ensure_shape(
                inputs,
                (None, img_dim, img_dim, self.in_channels))
        enc = self.init_conv(inputs)

        hiddens = []
        for downlayer, downsample in zip(self.downs, self.downsamples):
            enc = downlayer(enc)
            hiddens.append(enc)
            enc = downsample(enc)
        enc = self.downs[-1](enc)
        hiddens.append(enc)

        out = self.attn(enc)
        return out, hiddens

class MinUnetUp(tf.keras.Model):
    """
    MinUnetUp preps latent image and UnetDown hidden outputs to an image through a series of
    concatenatation, convolutions and up samples.
    only does double convolution and upsample. no attentions
    """
    def __init__(self, out_channels, params):
        super().__init__()

        self.params = params

        model_dim = params['model_dim']
        unet_dim = params['unet_dim']
        dim_mults = params['dim_mults']
        convnext_mult = params['convnext_mult']
        init_dim = params.get('init_dim', None)
        if init_dim is None:
            init_dim = unet_dim // 3 * 2

        self.dims = [init_dim] + [unet_dim * m for m in dim_mults]
        in_out = list(zip(self.dims[:-1], self.dims[1:]))
        mid_dim = self.dims[-1]

        self.attn = _conv_next_block(model_dim, mid_dim, mult=convnext_mult)
        self.ups = [
            _conv_next_block(dim_out * 2, dim_in, mult=convnext_mult)
            for dim_in, dim_out in in_out[::-1]
        ]
        self.upsamples = [
            tf.keras.layers.UpSampling2D(size=(2, 2))
            for _ in range(len(in_out)-1)
        ]
        self.final = tf.keras.layers.Conv2D(out_channels, 1)

    @tf.function
    def call(self, inputs, hiddens):
        """
        Model call implementation
        """

        nlayers = len(self.ups)
        img_dim = int(self.params['image_dim'] / (2 ** (nlayers-1)))
        inputs = tf.ensure_shape(
                inputs,
                (None, img_dim, img_dim, self.params['model_dim']))
        dec = self.attn(inputs)

        for i, (uplayer, upsample) in enumerate(zip(self.ups, self.upsamples)):
            dec = tf.ensure_shape(
                    dec,
                    (None, img_dim, img_dim, self.dims[-i-1]))
            if hiddens is None:
                hidden = dec
            else:
                hidden = tf.ensure_shape(
                        hiddens.read(nlayers-1-i),
                        (None, img_dim, img_dim, self.dims[-i-1]))
            dec = uplayer(tf.concat([dec, hidden], -1))
            dec = upsample(dec)
            img_dim *= 2
        dec = tf.ensure_shape(
                dec,
                (None, img_dim, img_dim, self.dims[-i-1]))
        if hiddens is None:
            hidden = dec
        else:
            hidden = tf.ensure_shape(
                    hiddens.read(0),
                    (None, img_dim, img_dim, self.dims[-i-1]))
        dec = self.ups[-1](tf.concat([dec, hidden], -1))

        out = self.final(dec)
        return out

class UnetDown(tf.keras.Model):
    """
    UnetDown preps image into latent image through a series of convolutions and down samples
    """
    def __init__(self, params):
        super().__init__()

        self.params = params

        model_dim = params['model_dim']
        unet_dim = params['unet_dim']
        dim_mults = params['dim_mults']
        convnext_mult = params['convnext_mult']
        init_dim = params.get('init_dim', None)
        if init_dim is None:
            init_dim = unet_dim // 3 * 2

        dims = [init_dim] + [unet_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        self.init_conv = tf.keras.layers.Conv2D(init_dim, 7, padding='same')
        self.downs = [
            tf.keras.Sequential([
                _conv_next_block(dim_in, dim_out, mult=convnext_mult),
                _conv_next_block(dim_out, dim_out, mult=convnext_mult),
                _linear_attention(dim_out, dim_out, params),
            ])
            for dim_in, dim_out in in_out
        ]
        self.downsamples = [
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
            for _ in range(len(in_out)-1)
        ]
        self.attn = _conv_next_block(mid_dim, model_dim, mult=convnext_mult)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ])
    def call(self, inputs):
        """
        Model call implementation
        """

        img_dim = self.params['image_dim']
        inputs = tf.ensure_shape(
                inputs,
                (None, img_dim, img_dim, self.in_channels))
        enc = self.init_conv(inputs)

        hiddens = []
        for downlayer, downsample in zip(self.downs, self.downsamples):
            enc = downlayer(enc)
            hiddens.append(enc)
            enc = downsample(enc)
        enc = self.downs[-1](enc)
        hiddens.append(enc)

        out = self.attn(enc)
        return out, hiddens

class UnetUp(tf.keras.Model):
    """
    UnetUp preps latent image and UnetDown hidden outputs to an image through a series of
    concatenatation, convolutions and up samples
    """
    def __init__(self, out_channels, params):
        super().__init__()

        self.params = params

        model_dim = params['model_dim']
        unet_dim = params['unet_dim']
        dim_mults = params['dim_mults']
        convnext_mult = params['convnext_mult']
        init_dim = params.get('init_dim', None)
        if init_dim is None:
            init_dim = unet_dim // 3 * 2

        dims = [init_dim] + [unet_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        self.attn = _conv_next_block(model_dim, mid_dim, mult=convnext_mult)
        self.ups = [
            tf.keras.Sequential([
                _conv_next_block(dim_out * 2, dim_in, mult=convnext_mult),
                _conv_next_block(dim_in, dim_in, mult=convnext_mult),
                _linear_attention(dim_in, dim_in, params),
            ])
            for dim_in, dim_out in in_out[::-1]
        ]
        self.upsamples = [
            tf.keras.layers.UpSampling2D(size=(2, 2))
            for _ in range(len(in_out)-1)
        ]
        self.final = tf.keras.Sequential([
            _conv_next_block(unet_dim, unet_dim, mult=convnext_mult),
            tf.keras.layers.Conv2D(out_channels, 1),
        ])

    @tf.function
    def call(self, inputs, hiddens):
        """
        Model call implementation
        """

        nlayers = len(self.ups)
        img_dim = int(self.params['image_dim'] / (2 ** (nlayers-1)))
        inputs = tf.ensure_shape(
                inputs,
                (None, img_dim, img_dim, self.params['model_dim']))
        dec = self.attn(inputs)

        for i, (uplayer, upsample) in enumerate(zip(self.ups, self.upsamples)):
            dec = tf.ensure_shape(
                    dec,
                    (None, img_dim, img_dim, self.dims[-i-1]))
            if hiddens is None:
                hidden = dec
            else:
                hidden = tf.ensure_shape(
                        hiddens.read(nlayers-1-i),
                        (None, img_dim, img_dim, self.dims[-i-1]))
            dec = uplayer(tf.concat([dec, hidden], -1))
            dec = upsample(dec)
            img_dim *= 2
        dec = tf.ensure_shape(
                dec,
                (None, img_dim, img_dim, self.dims[-i-1]))
        if hiddens is None:
            hidden = dec
        else:
            hidden = tf.ensure_shape(
                    hiddens.read(0),
                    (None, img_dim, img_dim, self.dims[-i-1]))
        dec = self.ups[-1](tf.concat([dec, hidden], -1))

        out = self.final(dec)
        return out
