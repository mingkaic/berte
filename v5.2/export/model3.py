#!/usr/bin/env python3
"""
This module includes unet specific models
"""
import os
import json

import tensorflow as tf

import common.models as models
import common.diffusion as diffs
import common.berts as berts
import common.readwrite as rw

import export.model as local_model

def _upsample(dim):
    return tf.keras.layers.Conv2DTranspose(dim, 4, strides=(2,2), padding='same')

def _downsample(dim):
    return tf.keras.layers.Conv2D(dim, 4, strides=(2,2), padding='same')

def _linear_attention(dim, params):
    return models.Multiplex([None, tf.keras.Sequential([
        tf.keras.layers.GroupNormalization(1),
        diffs.ConvAttention(dim,
            diffs.attention_init_builder().\
                num_heads(params['num_heads']).\
                attention_dim(params['attention_dim']).\
                strategy('linear_attention').build()),
    ])])

def _attention(dim, params):
    return models.Multiplex([None, tf.keras.Sequential([
        tf.keras.layers.GroupNormalization(1),
        diffs.ConvAttention(dim,
            diffs.attention_init_builder().\
                num_heads(params['num_heads']).\
                attention_dim(params['attention_dim']).\
                strategy('attention').build()),
    ])])

def _conv_next_block(dim_in, dim, mult=2, activation='gelu'):
    """
    https://arxiv.org/abs/2201.03545
    """
    return models.Multiplex([
        tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim_in, 7, padding='same', groups=dim_in),
            tf.keras.layers.GroupNormalization(1),
            tf.keras.layers.Conv2D(dim * mult, 3, padding='same', activation=activation),
            tf.keras.layers.GroupNormalization(1),
            tf.keras.layers.Conv2D(dim, 3, padding='same'),
        ]),
        tf.keras.layers.Conv2D(dim, 1, padding='valid') if dim_in != dim else None,
    ])

class TextEmbed(tf.keras.Model):
    def __init__(self, params):
        super().__init__(self)

        self.params = params

        model_dim = params['model_dim']

        self.embedder = berts.InputEmbed(model_dim,
                berts.input_embed_init_builder().\
                        maximum_position_encoding(params['max_pe']).\
                        vocab_size(params['vocab_size']).\
                        dropout_rate(params['dropout_rate']).build())
        self.encoder = berts.MaskedEncoder(params['num_enc_layers'],
                berts.encoder_init_builder().\
                        model_dim(model_dim).\
                        num_heads(params['num_heads']).\
                        dff(params['dff']).\
                        dropout_rate(params['dropout_rate']).\
                        use_bias(True).build())

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, inputs, training=False):
        # input.shape == (batch_size, x)
        # emb.shape == (batch_size, x, model_dim)
        emb = self.embedder(inputs, training)
        # enc.shape == (batch_size, x, model_dim)
        enc = self.encoder(emb, berts.create_padding_mask(inputs), training)
        return enc

    def get_config(self):
        config = super().get_config()
        config.update({ 'params': self.params })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def memory_rw(params):
    if params is None:
        return rw.KerasReadWriter('memory', local_model.Memory)

    return rw.KerasReadWriter('memory', local_model.Memory,
            params['num_mem_enc_layers'],
            params['num_mem_layers'],
            params['memory_dim'],
            berts.perceiver_init_builder().\
                    model_dim(params['model_dim']).\
                    num_heads(params['num_heads']).\
                    dff(params['dff']).\
                    dropout_rate(params['dropout_rate']).\
                    use_bias(True).build())

def text_extruder_rw(params):
    if params is None:
        return rw.KerasReadWriter('text_decoder', local_model.Perceiver)

    return rw.KerasReadWriter('text_decoder', local_model.Perceiver,
            params['num_perceiver_layers'],
            berts.perceiver_init_builder().\
                    model_dim(params['model_dim']).\
                    num_heads(params['num_heads']).\
                    dff(params['dff']).\
                    dropout_rate(params['dropout_rate']).\
                    use_bias(True).build())

class MemoryTextProcessor(rw.SaveableModule):
    """
    MemoryTextProcessor encoding latent space accepting texts.
    """
    def __init__(self, model_path, params):
        super().__init__(model_path, {
            'text_embedder': rw.KerasReadWriter('text_embedder', TextEmbed, params),
            'memory': memory_rw(params),
            'text_extruder': text_extruder_rw(params),
        })

        self.model_path = model_path
        self.params = params

        self.embedder = self.elems['text_embedder']
        self.mem = self.elems['memory']
        self.extruder = self.elems['text_extruder']

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, inputs, training=False):
        """
        Model call implementation
        """

        # input.shape == (batch_size, x)
        # emb.shape == (batch_size, x, model_dim)
        emb = self.embedder(inputs, training)
        # mem.shape == (batch_size, memory_dim, model_dim)
        mem = self.mem(emb, training)
        # out.shape == (batch_size, x, model_dim)
        out = self.extruder(mem, emb, training)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({ 'model_path': self.model_path, 'params': self.params })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ImageEmbed(tf.keras.Model):
    def __init__(self, params):
        super().__init__(self)

        self.params = params

        model_dim = params['model_dim']
        self.emb = tf.keras.Sequential([
            _attention(model_dim, params),
            tf.keras.layers.Reshape([-1, model_dim]),
        ])

    def call(self, inputs):
        return self.emb(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({ 'params': self.params })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ImageExtrude(tf.keras.Model):
    def __init__(self, params):
        super().__init__(self)

        self.params = params

        self.model_dim = params['model_dim']
        self.perceiver = berts.Perceiver(1, params)
        self.extrude = _attention(self.model_dim, params)

    def call(self, inputs, memory, orig_shape, training=False):
        batch_size = orig_shape[0]
        h_dim = orig_shape[1]
        w_dim = orig_shape[2]

        out = self.perceiver(memory, inputs, training)
        out = tf.reshape(out, [batch_size, h_dim, w_dim, self.model_dim])
        out = self.extrude(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({ 'params': self.params })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MemoryImageProcessor(rw.SaveableModule):
    """
    MemoryImageProcesso encoding latent space accepting images.
    """
    def __init__(self, model_path, params):
        super().__init__(self, model_path, {
            'image_embed': rw.KerasReadWriter('image_embed', ImageEmbed, {
                'model_dim': params['model_dim'],
                'num_heads': params['num_heads'],
                'attention_dim': params['attention_dim']}),
            'memory': rw.KerasReadWriter('memory', local_model.Memory,
                params['num_mem_enc_layers'],
                params['num_mem_layers'],
                params['memory_dim'],
                berts.perceiver_init_builder().\
                        model_dim(params['model_dim']).\
                        num_heads(params['num_heads']).\
                        dff(params['dff']).\
                        dropout_rate(params['dropout_rate']).\
                        use_bias(True).build()),
            'image_extrude': rw.KerasReadWriter('image_extrude', ImageExtrude, {
                'model_dim': params['model_dim'],
                'num_heads': params['num_heads'],
                'attention_dim': params['attention_dim']}),
        })

        self.emb = self.elems['image_embed']
        self.mem = self.elems['memory']
        self.ext = self.elems['image_extrude']

    def call(self, inputs, training=False):
        """
        Model call implementation
        """

        # inputs.shape == (batch_size, h_dim, w_dim, model_dim)
        # emb.shape == (batch_size, h_dim * w_dim, model_dim)
        emb = self.emb(inputs)
        # mem.shape == (batch_size, memory_dim, model_dim)
        mem = self.memory(emb, training)
        # ext.shape == (batch_size, h_dim, w_dim, model_dim)
        ext = self.ext(mem, emb, tf.shape(inputs), training)
        return ext

    def get_config(self):
        config = super().get_config()
        config.update({ 'model_path': self.model_path, 'params': self.params })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def tokenizer_rw(tokenizer_setup, metadata_path):
    if tokenizer_setup is None:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as file:
                content = file.read()
                setup = json.loads(content)
                tokenizer_setup = {
                    'add_bos': setup.get('add_bos', False),
                    'add_eos': setup.get('add_eos', False),
                }
        else:
            tokenizer_setup = {
                'add_bos': False,
                'add_eos': False,
            }
    return rw.TokenizerReadWriter('tokenizer', tokenizer_setup)

def text_predictor_rw(params):
    if params is None:
        return rw.KerasReadWriter('text_predictor', local_model.Predictor)

    return rw.KerasReadWriter('text_predictor', local_model.Predictor,
            params['model_dim'], params['vocab_size'])

class TextBert(rw.SaveableModule):
    """
    TextBert handles text transformation
    """
    def __init__(self, model_path, optimizer, params, tokenizer_setup):
        super().__init__(model_path, {
            'text_metadata': rw.FileReadWriter('text_metadata'),
            'tokenizer': tokenizer_rw(tokenizer_setup, os.path.join(model_path, 'text_metadata')),
            'memory_processor': rw.SModuleReadWriter('memory_processor',
                MemoryTextProcessor, params),
            'text_predictor': text_predictor_rw(params),
        })

        self.metadata = self.elems['text_metadata']
        self.tokenizer = self.elems['tokenizer']
        self.processor = self.elems['memory_processor']
        self.predictor = self.elems['text_predictor']

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        out = self.tokenizer.tokenize(sentence)
        if not isinstance(out, tf.Tensor):
            out = out.to_tensor()
        return out

    def predict(self, tokens, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        enc = self.processor(tokens, training=training)
        pred, debug_info = self.predictor(enc)
        debug_info.update({'prediction.prediction': pred})
        return (pred, debug_info)

class UnetDown(tf.keras.Model):
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
                _linear_attention(dim_out, params),
            ])
            for dim_in, dim_out in in_out
        ]
        self.downsamples = [
            _downsample(dim_out)
            for _, dim_out in in_out[:-1]
        ]
        self.attn = _conv_next_block(mid_dim, model_dim, mult=convnext_mult)

    def call(self, inputs, training=False):
        """
        Layer call implementation
        """

        enc = self.init_conv(inputs)

        hiddens = []
        for (downlayer, downsample) in zip(self.downs, self.downsamples):
            enc = downlayer(enc)
            hiddens.append(enc)
            enc = downsample(enc)
        enc = self.downs[-1](enc)
        hiddens.append(enc)

        out = self.attn(enc)
        return out, hiddens

    def get_config(self):
        config = super().get_config()
        config.update({
            'params': self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def unet_down_rw(params):
    if params is None:
        return rw.KerasReadWriter('image_unetd', UnetDown)

    return rw.KerasReadWriter('image_unetd', UnetDown,
            berts.perceiver_init_builder().\
                    model_dim(params['model_dim']).\
                    num_heads(params['num_heads']).\
                    dff(params['dff']).\
                    dropout_rate(params['dropout_rate']).\
                    use_bias(True).build())

class ImageUnet(rw.SaveableModule):
    def __init__(
        self,
        dim,
        params,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        convnext_mult=2,
    ):
        super().__init__()

        if init_dim is None:
            init_dim = dim // 3 * 2

        if out_dim is None:
            out_dim = channels

        self.dim = dim
        self.params = params
        self.init_dim = init_dim
        self.out_dim = out_dim
        self.dim_mults = dim_mults
        self.channels = channels
        self.convnext_mult = convnext_mult

        dims = [init_dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        self.init_conv = tf.keras.layers.Conv2D(init_dim, 7, padding='same')

        self.downs = [
            tf.keras.Sequential([
                _conv_next_block(dim_in, dim_out, mult=convnext_mult),
                _conv_next_block(dim_out, dim_out, mult=convnext_mult),
                _linear_attention(dim_out, params),
            ])
            for dim_in, dim_out in in_out
        ]
        self.downsamples = [
            _downsample(dim_out)
            for _, dim_out in in_out[:-1]
        ]
        model_dim = params['model_dim']
        self.mid_block1 = _conv_next_block(mid_dim, model_dim, mult=convnext_mult)

        self.mem = MemoryImageProcessor('', params)

        self.mid_block2 = _conv_next_block(model_dim, mid_dim, mult=convnext_mult)
        self.ups = [
            tf.keras.Sequential([
                _conv_next_block(dim_out * 2, dim_in, mult=convnext_mult),
                _conv_next_block(dim_in, dim_in, mult=convnext_mult),
                _linear_attention(dim_in, params),
                _upsample(dim_in),
            ])
            for dim_in, dim_out in in_out[:0:-1]
        ]

        self.final_conv = tf.keras.Sequential([
            _conv_next_block(dim, dim, mult=convnext_mult),
            tf.keras.layers.Conv2D(out_dim, 1),
        ])

    def call(self, inputs, training=False):
        """
        Layer call implementation
        """

        inputs = tf.ensure_shape(inputs, [None, None, None, self.channels])

        enc = self.init_conv(inputs)

        hiddens = []
        for (downlayer, downsample) in zip(self.downs, self.downsamples):
            enc = downlayer(enc)
            hiddens.append(enc)
            enc = downsample(enc)
        enc = self.downs[-1](enc)
        hiddens.append(enc)

        mem = self.mid_block1(enc)
        mem = self.mem(mem, training)
        dec = self.mid_block2(mem)

        for uplayer in self.ups:
            dec = uplayer(tf.concat([dec, hiddens.pop()], -1))

        out = self.final_conv(dec)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim':           self.dim,
            'params':        self.params,
            'init_dim':      self.init_dim,
            'out_dim':       self.out_dim,
            'dim_mults':     self.dim_mults,
            'channels':      self.channels,
            'convnext_mult': self.convnext_mult,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
