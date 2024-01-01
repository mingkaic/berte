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

    def call(self, inputs, latent, orig_shape, training=False):
        batch_size = orig_shape[0]
        h_dim = orig_shape[1]
        w_dim = orig_shape[2]

        out = self.perceiver(inputs, latent, training)
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
        super().__init__(model_path, {
            'image_embed': rw.KerasReadWriter('image_embed', ImageEmbed, {
                'model_dim': params['model_dim'],
                'num_heads': params['num_heads'],
                'attention_dim': params['attention_dim'],
            }),
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
                'attention_dim': params['attention_dim'],
                'dff': params['dff'],
                'dropout_rate': params['dropout_rate'],
                'use_bias': True,
            }),
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
        mem = self.mem(emb, training)
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
            'text_metadata': rw.FileReadWriter('text_metadata',
                lambda metadata: metadata.update({
                    'optimizer_iter': int(optimizer.iterations.numpy()),
                }) if optimizer is not None else None),
            'tokenizer': tokenizer_rw(tokenizer_setup, os.path.join(model_path, 'text_metadata')),
            'memory_processor': rw.SModuleReadWriter('memory_processor',
                MemoryTextProcessor, params),
            'text_predictor': text_predictor_rw(params),
        })

        self.model_path = model_path
        self.optimizer = optimizer # todo use
        self.params = params
        self.tokenizer_setup = tokenizer_setup

        self.metadata = self.elems['text_metadata']
        self.tokenizer = self.elems['tokenizer']
        self.processor = self.elems['memory_processor']
        self.predictor = self.elems['text_predictor']

        if optimizer is not None:
            optimizer.iterations.assign(int(self.metadata.get('optimizer_iter', 0)))

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        out = self.tokenizer.tokenize(sentence)
        if not isinstance(out, tf.Tensor):
            out = out.to_tensor()
        return out

    def call(self, inputs, training=False):
        """
        Layer call implementation
        """

        enc = self.processor(inputs, training=training)
        pred, debug_info = self.predictor(enc)
        debug_info.update({'prediction.prediction': pred})
        return (pred, debug_info)

    def predict(self, tokens, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        return self.call(tokens, training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_path': self.model_path,
            'optimizer': self.optimizer,
            'params': self.params,
            'tokenizer_setup': self.tokenizer_setup,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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

    def call(self, inputs):
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

class UnetUp(tf.keras.Model):
    def __init__(self, out_dim, params):
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
                _linear_attention(dim_in, params),
                _upsample(dim_in),
            ])
            for dim_in, dim_out in in_out[:0:-1]
        ]
        self.final = tf.keras.Sequential([
            _conv_next_block(unet_dim, unet_dim, mult=convnext_mult),
            tf.keras.layers.Conv2D(out_dim, 1),
        ])

    def call(self, inputs, hiddens):
        """
        Layer call implementation
        """

        dec = self.attn(inputs)

        for uplayer in self.ups:
            dec = uplayer(tf.concat([dec, hiddens.pop()], -1))

        out = self.final(dec)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'params': self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ImageUnet(rw.SaveableModule):
    def __init__(
        self,
        model_path,
        out_dim,
        optimizer,
        params,
    ):
        super().__init__(model_path, {
            'image_metadata': rw.FileReadWriter('image_metadata',
                lambda metadata: metadata.update({
                    'optimizer_iter': int(optimizer.iterations.numpy()),
                }) if optimizer is not None else None),
            'image_unetd': rw.KerasReadWriter('image_unetd', UnetDown, params),
            'memory_processor': rw.SModuleReadWriter('memory_processor',
                MemoryImageProcessor, params),
            'image_unetu': rw.KerasReadWriter('image_unetu', UnetUp, out_dim, params),
        })

        self.model_path = model_path
        self.out_dim = out_dim
        self.optimizer = optimizer
        self.params = params

        self.metadata = self.elems['image_metadata']
        self.unetd = self.elems['image_unetd']
        self.mem = self.elems['memory_processor']
        self.unetu = self.elems['image_unetu']

        if optimizer is not None:
            optimizer.iterations.assign(int(self.metadata.get('optimizer_iter', 0)))

    def call(self, inputs, training=False):
        """
        Layer call implementation
        """

        enc, hiddens = self.unetd(inputs)
        mem = self.mem(enc, training)
        dec = self.unetu(mem, hiddens)
        return dec

    def predict(self, inputs, training=False):

        return self.call(inputs, training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_path': self.model_path,
            'out_dim':    self.out_dim,
            'optimizer': self.optimizer,
            'params':     self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
