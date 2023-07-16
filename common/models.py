#!/usr/bin/env python3
"""
This module specifies common model
"""
import tensorflow as tf

def _scaled_dot_product_attention(query, key, value, mask):
    """
    Calculate the attention weights.
    query, key, value must have matching leading dimensions.
    key, value must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    query: shape == (..., seq_len_q, depth)
    key:   shape == (..., seq_len_kv, depth)
    value: shape == (..., seq_len_kv, depth)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(query, key, transpose_b=True)  # (..., seq_len_q, seq_len_kv)

    # scale matmul_qk
    dkey = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dkey)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_kv) so that the scores
    # add up to 1.
    # (..., seq_len_q, seq_len_kv)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, value)  # (..., seq_len_q, depth)

    return output

class SwiGLU(tf.keras.layers.Layer):
    """
    SwiGLU activation: Swish(xW+b, beta) * (xV+b)
    """
    def __init__(self, in_size, beta=1.0, use_bias=True):
        super().__init__()
        self.in_size = in_size
        self.beta = beta
        self.use_bias = use_bias

        self.dense = tf.keras.layers.Dense(in_size * 2, use_bias=use_bias)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ])
    def call(self, inputs):
        inputs = tf.ensure_shape(inputs, [None, None, self.in_size])
        tmp = self.dense(inputs) # shape=[None, None, 2*in_size]
        unsilued = tmp[:, :, :self.in_size]
        silued = tmp[:, :, self.in_size:]
        return unsilued * tf.nn.silu(silued, beta=self.beta)

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_size': self.in_size,
            'beta': self.beta,
            'use_bias': self.use_bias,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class UnmaskedMultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention layer used in Encoders and Decoders
    """
    def __init__(self, model_dim, num_heads, use_bias=True):
        super().__init__()

        assert model_dim % num_heads == 0
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        self.wquery = tf.keras.layers.Dense(model_dim, use_bias=use_bias)
        self.wkey = tf.keras.layers.Dense(model_dim, use_bias=use_bias)
        self.wvalue = tf.keras.layers.Dense(model_dim, use_bias=use_bias)
        self.dense = tf.keras.layers.Dense(model_dim, use_bias=use_bias)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ])
    def call(self, query, key, value):
        batch_size = tf.shape(query)[0]

        query = tf.ensure_shape(query, [None, None, self.model_dim])
        key   = tf.ensure_shape(key,   [None, None, self.model_dim])
        value = tf.ensure_shape(value, [None, None, self.model_dim])

        query = self.wquery(query) # (batch_size, seq_len_q, model_dim)
        key   = self.wkey(key)     # (batch_size, seq_len_k, model_dim)
        value = self.wvalue(value) # (batch_size, seq_len_v, model_dim)

        # inputs.X.shape == (batch_size, num_heads, seq_len_X, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        scaled_attention = _scaled_dot_product_attention(
            query=self.split_heads(query, batch_size),
            key=self.split_heads(key, batch_size),
            value=self.split_heads(value, batch_size),
            mask=None)

        # scaled_attention.shape == (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concat_attention.shape == (batch_size, seq_len_q, model_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim))
        output = self.dense(concat_attention) # (batch_size, seq_len_q, model_dim)

        return output

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
    ])
    def split_heads(self, inp, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        depth = self.model_dim // self.num_heads
        inp = tf.reshape(inp, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(inp, perm=[0, 2, 1, 3])

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'use_bias': self.use_bias,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MaskedMultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention layer used in Encoders and Decoders
    """
    def __init__(self, model_dim, num_heads, use_bias=True):
        super().__init__()

        assert model_dim % num_heads == 0
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        self.wquery = tf.keras.layers.Dense(model_dim, use_bias=use_bias)
        self.wkey = tf.keras.layers.Dense(model_dim, use_bias=use_bias)
        self.wvalue = tf.keras.layers.Dense(model_dim, use_bias=use_bias)
        self.dense = tf.keras.layers.Dense(model_dim, use_bias=use_bias)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ])
    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]

        query = tf.ensure_shape(query, [None, None, self.model_dim])
        key   = tf.ensure_shape(key,   [None, None, self.model_dim])
        value = tf.ensure_shape(value, [None, None, self.model_dim])

        query = self.wquery(query) # (batch_size, seq_len_q, model_dim)
        key   = self.wkey(key)     # (batch_size, seq_len_k, model_dim)
        value = self.wvalue(value) # (batch_size, seq_len_v, model_dim)

        # inputs.X.shape == (batch_size, num_heads, seq_len_X, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        scaled_attention = _scaled_dot_product_attention(
            query=self.split_heads(query, batch_size),
            key=self.split_heads(key, batch_size),
            value=self.split_heads(value, batch_size),
            mask=mask)

        # scaled_attention.shape == (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concat_attention.shape == (batch_size, seq_len_q, model_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim))
        output = self.dense(concat_attention) # (batch_size, seq_len_q, model_dim)

        return output

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
    ])
    def split_heads(self, inp, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        depth = self.model_dim // self.num_heads
        inp = tf.reshape(inp, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(inp, perm=[0, 2, 1, 3])

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'use_bias': self.use_bias,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
