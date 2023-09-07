#!/usr/bin/env python3
"""
This module includes functions that help build a perceiver
"""

import numpy as np

import tensorflow as tf

import common.models as models
from common.builder import Builder

def create_padding_mask(seq):
    """
    create mask for encoder and transformer
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def input_embed_init_builder():
    """ create Builder for input_embed init """
    return Builder(['maximum_position_encoding', 'vocab_size', 'dropout_rate'],
        default_values={'dropout_rate': 0.1})

def perceiver_init_builder():
    """ create Builder for perceiver init """
    return Builder(['model_dim', 'num_heads', 'dff', 'dropout_rate', 'use_bias'],
        default_values={'dropout_rate': 0.1, 'use_bias': False})

def encoder_init_builder():
    """ encapsulate params """
    return Builder(['model_dim', 'num_heads', 'dff', 'dropout_rate', 'use_bias'],
        default_values={'dropout_rate': 0.1, 'use_bias': False})

class InputEmbed(tf.keras.layers.Layer):
    """
    Embed input with positional encoding
    """
    def __init__(self, model_dim, params):
        super().__init__()

        vocab_size = params['vocab_size']
        dropout_rate = params['dropout_rate']
        maximum_position_encoding = params['maximum_position_encoding']

        self.model_dim = model_dim
        self.params = params
        self.pos_encoding = _positional_encoding(maximum_position_encoding, model_dim)
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, inp, training):
        """
        Layer call implementation
        """
        # inp.shape == (batch_size, input_seq_len)
        seq_len = tf.shape(inp)[1]

        inp = self.embedding(inp) # shape == (batch_size, input_seq_len, d_model)
        inp *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        inp += self.pos_encoding[:, :seq_len, :]
        return self.dropout(inp, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_dim':  self.model_dim,
            'params': self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Perceiver(tf.keras.layers.Layer):
    """
    Encapsulate perceiver layers using improvements as noted in LLaMa
    https://arxiv.org/pdf/2302.13971.pdf
    """
    def __init__(self, num_layers, params):
        super().__init__()
        self.params = params
        self.perceiver_layers = [_perceiverLayer(params) for _ in range(num_layers)]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, enc, latent, training):
        """
        Layer call implementation
        """
        # enc.shape == (batch_size, ?, ?)
        # latent.shape == (batch_size, latent_dim, model_dim)
        for perceiver_layer in self.perceiver_layers:
            # enc.shape == (batch_size, latent_dim, model_dim)
            enc = perceiver_layer(enc, latent, training=training)
        return enc

    def multi_call(self, enc, latent, training=False, *args):
        """
        calls with multiple latents
        """
        # enc.shape == (batch_size, ?, ?)
        # latent.shape == (batch_size, latent_dim, model_dim)
        for perceiver_layer in self.perceiver_layers:
            # enc.shape == (batch_size, latent_dim, model_dim)
            enc = perceiver_layer(enc, latent, training=training)
            for lat in args:
                enc = perceiver_layer(enc, lat, training=training)
        return enc

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': len(self.perceiver_layers),
            'params': self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Encoder(tf.keras.layers.Layer):
    """
    Encapsulate encoder layers using improvements as noted in LLaMa
    """
    def __init__(self, num_layers, params):
        assert num_layers > 0
        super().__init__()
        self.params = params
        self.enc_layers = [_encoderLayer(params) for _ in range(num_layers)]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, enc, training):
        """
        Layer call implementation
        """
        # enc.shape == (batch_size, ?, ?)
        for enc_layer in self.enc_layers:
            # enc.shape == (batch_size, enc_seq_len, model_dim)
            enc = enc_layer(enc, training)
        return enc

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': len(self.enc_layers),
            'params': self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MaskedEncoder(tf.keras.layers.Layer):
    """
    Encapsulate encoder layers using improvements as noted in LLaMa with masks
    """
    def __init__(self, num_layers, params):
        assert num_layers > 0
        super().__init__()
        self.params = params
        self.masked_enc_layer = _maskedEncoderLayer(params)
        self.enc_layers = [_encoderLayer(params) for _ in range(num_layers-1)]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, enc, mask, training):
        """
        Layer call implementation
        """
        # enc.shape == (batch_size, ?, ?)
        enc = self.masked_enc_layer(enc, mask, training)
        for enc_layer in self.enc_layers:
            # enc.shape == (batch_size, enc_seq_len, model_dim)
            enc = enc_layer(enc, training)
        return enc

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': len(self.enc_layers)+1,
            'params': self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def _get_angles(pos, i, model_dim):
    """
    position encoding helper
    """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(model_dim))
    return pos * angle_rates

def _positional_encoding(position, model_dim):
    """
    encode token position
    """
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis],
            np.arange(model_dim)[np.newaxis, :], model_dim)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, tf.float32)

class _perceiverLayer(tf.keras.layers.Layer):
    """
    _perceiverLayer presents a single encoder layer in Conceiver with prenormalization and SwiGLU
    """
    def __init__(self, params):
        super().__init__()
        model_dim = params['model_dim']
        num_heads = params['num_heads']
        dff = params['dff']
        dropout_rate = params['dropout_rate']
        use_bias = params['use_bias']

        self.params = params
        self.mha = models.UnmaskedMultiHeadAttention(model_dim, num_heads, use_bias=use_bias)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, use_bias=use_bias),
            models.SwiGLU(dff, use_bias=use_bias),
            tf.keras.layers.Dense(model_dim, use_bias=use_bias)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, inp, latent, training):
        """
        Layer call implementation
        """
        # inp.shape == (batch_size, ?, model_dim)
        # latent.shape == (batch_size, latent_dim, model_dim)
        inp = tf.ensure_shape(inp, [None, None, self.params['model_dim']])
        latent = tf.ensure_shape(latent, [None, None, self.params['model_dim']])

        norm_inp = self.layernorm1(inp)
        norm_latent = self.layernorm2(latent)
        # attn_output.shape == (batch_size, latent_dim, model_dim)
        attn_output = self.mha(query=norm_latent, key=norm_inp, value=norm_inp)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = latent + attn_output

        ffn_output = self.ffn(self.layernorm3(out1)) # (batch_size, latent_dim, model_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output  # (batch_size, latent_dim, model_dim)
        return out2

    def get_config(self):
        config = super().get_config()
        config.update({'params': self.params})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class _encoderLayer(tf.keras.layers.Layer):
    """
    _encoderLayer presents a single encoder layer in BERT from LLaMa
    """
    def __init__(self, params):
        super().__init__()
        model_dim = params['model_dim']
        num_heads = params['num_heads']
        dff = params['dff']
        dropout_rate = params['dropout_rate']
        use_bias = params['use_bias']

        self.params = params
        self.mha = models.UnmaskedMultiHeadAttention(model_dim, num_heads, use_bias=use_bias)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, use_bias=use_bias),
            models.SwiGLU(dff, use_bias=use_bias),
            tf.keras.layers.Dense(model_dim, use_bias=use_bias),
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, inp, training):
        """
        Layer call implementation
        """
        # inp.shape == (batch_size, ?, model_dim)
        inp = tf.ensure_shape(inp, [None, None, self.params['model_dim']])

        norm_inp = self.layernorm1(inp)
        attn_output = self.mha(
            query=norm_inp,
            key=norm_inp,
            value=norm_inp)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inp + attn_output # shape == (batch_size, ?, model_dim)

        norm_out1 = self.layernorm2(out1)
        ffn_output = self.ffn(norm_out1) # shape == (batch_size, ?, model_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output # shape == (batch_size, ?, model_dim)
        return out2

    def get_config(self):
        config = super().get_config()
        config.update({'params': self.params})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class _maskedEncoderLayer(tf.keras.layers.Layer):
    """
    _encoderLayer presents a single encoder layer in BERT from LLaMa
    """
    def __init__(self, params):
        super().__init__()
        model_dim = params['model_dim']
        num_heads = params['num_heads']
        dff = params['dff']
        dropout_rate = params['dropout_rate']
        use_bias = params['use_bias']

        self.params = params
        self.mha = models.MaskedMultiHeadAttention(model_dim, num_heads, use_bias=use_bias)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, use_bias=use_bias),
            models.SwiGLU(dff, use_bias=use_bias),
            tf.keras.layers.Dense(model_dim, use_bias=use_bias),
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, inp, mask, training):
        """
        Layer call implementation
        """
        # inp.shape == (batch_size, ?, model_dim)
        inp = tf.ensure_shape(inp, [None, None, self.params['model_dim']])

        norm_inp = self.layernorm1(inp)
        attn_output = self.mha(
            query=norm_inp,
            key=norm_inp,
            value=norm_inp,
            mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inp + attn_output # shape == (batch_size, ?, model_dim)

        norm_out1 = self.layernorm2(out1)
        ffn_output = self.ffn(norm_out1) # shape == (batch_size, ?, model_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output # shape == (batch_size, ?, model_dim)
        return out2

    def get_config(self):
        config = super().get_config()
        config.update({'params': self.params})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
