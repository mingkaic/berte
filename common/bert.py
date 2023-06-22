#!/usr/bin/env python3
"""
This module includes functions that help build a perceiver
"""

import numpy as np

import tensorflow as tf

def create_padding_mask(seq):
    """
    create mask for encoder and transformer
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def perceiver_layer_init_param(model_dim, num_heads, dff, dropout_rate=0.1, use_bias=False):
    """ encapsulate params """
    return {
        "model_dim":    model_dim,
        "num_heads":    num_heads,
        "dff":          dff,
        "dropout_rate": dropout_rate,
        "use_bias":     use_bias
    }

def encoder_layer_init_param(model_dim, num_heads, dff, dropout_rate=0.1, use_bias=False):
    """ encapsulate params """
    return {
        "model_dim":    model_dim,
        "num_heads":    num_heads,
        "dff":          dff,
        "dropout_rate": dropout_rate,
        "use_bias":     use_bias
    }

def input_embed_call_param(inp, training=False):
    """ encapsulate params """
    return {
        "inp":      inp,
        "training": training,
    }

def perceiver_call_param(enc, latent, mask=None, training=False):
    """ encapsulate params """
    return {
        "enc":      enc,
        "latent":   latent,
        "mask":     mask,
        "training": training,
    }

def encoder_call_param(enc, mask=None, training=False):
    """ encapsulate params """
    return {
        "enc":      enc,
        "mask":     mask,
        "training": training,
    }

class InputEmbed(tf.keras.layers.Layer):
    """
    Embed input with positional encoding
    """
    def __init__(self, model_dim, maximum_position_encoding, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.model_dim = model_dim
        self.pos_encoding = _positional_encoding(maximum_position_encoding, model_dim)
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, *args, **kwargs):
        """
        Layer call implementation
        """
        inp = inputs["inp"]
        training = inputs["training"]

        # inp.shape == (batch_size, input_seq_len)
        seq_len = tf.shape(inp)[1]

        inp = self.embedding(inp) # shape == (batch_size, input_seq_len, d_model)
        inp *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        inp += self.pos_encoding[:, :seq_len, :]
        return self.dropout(inp, training=training)

class Perceiver(tf.keras.layers.Layer):
    """
    Encapsulate perceiver layers using improvements as noted in LLaMa
    https://arxiv.org/pdf/2302.13971.pdf
    """
    def __init__(self, num_layers, params):
        super().__init__()

        self.perceiver_layers = [_perceiverLayer(params) for _ in range(num_layers)]

    def call(self, inputs, *args, **kwargs):
        """
        Layer call implementation
        """
        enc = inputs["enc"]
        latent = inputs["latent"]
        mask = inputs["mask"]
        training = inputs["training"]

        # enc.shape == (batch_size, ?, ?)
        # latent.shape == (batch_size, latent_dim, model_dim)
        attention_weights = {}
        for i, perceiver_layer in enumerate(self.perceiver_layers):
            # enc.shape == (batch_size, latent_dim, model_dim)
            enc, block = perceiver_layer(_perceiver_layer_call_param(
                    enc, latent, mask=mask, training=training))
            mask = None

            attention_weights['perceiver_layer{}_block'.format(i+1)] = block

        return enc, attention_weights

class Encoder(tf.keras.layers.Layer):
    """
    Encapsulate encoder layers using improvements as noted in LLaMa
    """
    def __init__(self, num_layers, params):
        super().__init__()

        self.enc_layers = [_encoderLayer(params) for _ in range(num_layers)]

    def call(self, inputs, *args, **kwargs):
        """
        Layer call implementation
        """
        enc = inputs["enc"]
        mask = inputs["mask"]
        training = inputs["training"]

        # enc.shape == (batch_size, ?, ?)
        attention_weights = {}
        for i, enc_layer in enumerate(self.enc_layers):
            # enc.shape == (batch_size, enc_seq_len, model_dim)
            enc, block = enc_layer(_encoder_layer_call_param(enc, mask=mask, training=training))
            mask = None

            attention_weights['encoder_layer{}_block'.format(i+1)] = block

        return enc, attention_weights

def _mha_call_param(query, key, value, mask=None):
    """ mha_call_param encapsulates parameters passed into MHA """
    return {
        "query": query,
        "key": key,
        "value": value,
        "mask": mask,
    }

def _perceiver_layer_call_param(inp, latent, mask=None, training=False):
    return {
        "inp":      inp,
        "latent":   latent,
        "mask":     mask,
        "training": training,
    }

def _encoder_layer_call_param(inp, mask=None, training=False):
    return {
        "inp":      inp,
        "mask":     mask,
        "training": training,
    }

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

def _scaled_dot_product_attention(args):
    """Calculate the attention weights.
    query, key, value must have matching leading dimensions.
    key, value must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    args: dict with the following key-values
        query: shape == (..., seq_len_q, depth)
        key:   shape == (..., seq_len_kv, depth)
        value: shape == (..., seq_len_kv, depth)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    query = args["query"]
    key = args["key"]
    value = args["value"]
    mask = args["mask"]

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

    return output, attention_weights

class _multiHeadAttention(tf.keras.layers.Layer):
    """
    _multiHeadAttention layer used in Encoders and Decoders
    """
    def __init__(self, model_dim, num_heads, use_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim

        assert model_dim % self.num_heads == 0

        self.depth = model_dim // self.num_heads

        self.wquery = tf.keras.layers.Dense(model_dim, use_bias=use_bias)
        self.wkey = tf.keras.layers.Dense(model_dim, use_bias=use_bias)
        self.wvalue = tf.keras.layers.Dense(model_dim, use_bias=use_bias)

        self.dense = tf.keras.layers.Dense(model_dim, use_bias=use_bias)

    def split_heads(self, inp, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        inp = tf.reshape(inp, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inp, perm=[0, 2, 1, 3])

    def call(self, inputs, *args, **kwargs):
        """
        Layer call implementation
        """
        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]

        batch_size = tf.shape(query)[0]
        query = self.wquery(query) # (batch_size, seq_len_q, model_dim)
        key = self.wkey(key)       # (batch_size, seq_len_k, model_dim)
        value = self.wvalue(value) # (batch_size, seq_len_v, model_dim)

        # inputs.X.shape == (batch_size, num_heads, seq_len_X, depth)
        inputs = {
            "query": self.split_heads(query, batch_size),
            "key": self.split_heads(key, batch_size),
            "value": self.split_heads(value, batch_size),
            "mask": inputs["mask"],
        }

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = _scaled_dot_product_attention(inputs)

        # scaled_attention.shape == (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concat_attention.shape == (batch_size, seq_len_q, model_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim))

        output = self.dense(concat_attention) # (batch_size, seq_len_q, model_dim)

        return output, attention_weights

class _swiGLU(tf.keras.layers.Layer):
    """
    _swiGLU activation: Swish(xW+b, beta) * (xV+b)
    """
    def __init__(self, in_size, beta=1.0, use_bias=True):
        super().__init__()
        self.dense = tf.keras.layers.Dense(in_size * 2, use_bias=use_bias)
        self.in_size = in_size
        self.beta = beta

    def call(self, inputs, *args, **kwargs):
        """
        Layer call implementation
        """
        tmp = self.dense(inputs)
        shape = inputs.shape
        rank = len(shape)
        unsilued = tf.slice(tmp, [0] * rank, shape)
        silued = tf.slice(tmp, [0] * (rank - 1) + [self.in_size], shape)
        return unsilued * tf.nn.silu(silued, beta=self.beta)

class _perceiverLayer(tf.keras.layers.Layer):
    """
    _perceiverLayer presents a single encoder layer in Conceiver with prenormalization and SwiGLU
    """
    def __init__(self, params):
        super().__init__()

        model_dim = params["model_dim"]
        num_heads = params["num_heads"]
        dff = params["dff"]
        dropout_rate = params["dropout_rate"]
        use_bias = params["use_bias"]

        self.mha = _multiHeadAttention(model_dim, num_heads, use_bias=use_bias)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, use_bias=use_bias), # shape == (batch_size, ?, dff)
            _swiGLU(dff, use_bias=use_bias),
            tf.keras.layers.Dense(model_dim, use_bias=use_bias) # shape == (batch_size, ?, model_dim)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, *args, **kwargs):
        """
        Layer call implementation
        """
        inp = inputs["inp"]
        latent = inputs["latent"]
        training = inputs["training"]

        # inp.shape == (batch_size, ?, model_dim)
        # latent.shape == (batch_size, latent_dim, model_dim)
        norm_inp = self.layernorm1(inp)
        norm_latent = self.layernorm2(latent)
        # attn_output.shape == (batch_size, latent_dim, model_dim)
        # attn_weight_block.shape == (batch_size, num_heads, latent_dim, ?)
        attn_output, attn_weights_block = self.mha(
                _mha_call_param(
                    query=norm_latent,
                    key=norm_inp,
                    value=norm_inp,
                    mask=inputs["mask"]), *args, **kwargs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = latent + attn_output

        ffn_output = self.ffn(self.layernorm3(out1)) # (batch_size, latent_dim, model_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output  # (batch_size, latent_dim, model_dim)

        return out2, attn_weights_block

class _encoderLayer(tf.keras.layers.Layer):
    """
    _encoderLayer presents a single encoder layer in BERT from LLaMa
    """
    def __init__(self, params):
        super().__init__()

        model_dim = params["model_dim"]
        num_heads = params["num_heads"]
        dff = params["dff"]
        dropout_rate = params["dropout_rate"]
        use_bias = params["use_bias"]

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mha = _multiHeadAttention(model_dim, num_heads, use_bias=use_bias)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, use_bias=use_bias), # (batch_size, seq_len, dff)
            _swiGLU(dff, use_bias=use_bias),
            tf.keras.layers.Dense(model_dim, use_bias=use_bias) # (batch_size, seq_len, model_dim)
        ])

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, *args, **kwargs):
        """
        Layer call implementation
        """
        inp = inputs["inp"]
        training = inputs["training"]

        # inp.shape == (batch_size, ?, model_dim)
        norm_inp = self.layernorm1(inp)
        attn_output, attn_weights_block = self.mha(
                _mha_call_param(
                    query=norm_inp,
                    key=norm_inp,
                    value=norm_inp,
                    mask=inputs["mask"]), *args, **kwargs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inp + attn_output # shape == (batch_size, ?, model_dim)

        norm_out1 = self.layernorm2(out1)
        ffn_output = self.ffn(norm_out1) # shape == (batch_size, ?, model_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output # shape == (batch_size, ?, model_dim)

        return out2, attn_weights_block