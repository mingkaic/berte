#!/usr/bin/env python3
"""
This module includes unet specific models
"""
import os
import json
from packaging import version

import tensorflow as tf

import common.models as models
import common.diffusion as diffs
import common.berts as berts
import common.readwrite as rw

import export.model as local_model

if version.parse(tf.__version__) < version.parse('2.11.0'):
    import tensorflow_addons as tfa
    GroupNorm = tfa.layers.GroupNormalization
else:
    GroupNorm = tf.keras.layers.GroupNormalization

def _downsample(dim):
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

def _upsample(dim):
    return tf.keras.layers.UpSampling2D(size=(2, 2))

def _attention(dim, channel_dim, params):
    return models.Multiplex([None, tf.keras.Sequential([
        GroupNorm(1),
        diffs.ConvAttention(dim, channel_dim,
            diffs.attention_init_builder().\
                num_heads(params['num_heads']).\
                attention_dim(params['attention_dim']).\
                strategy('attention').build()),
    ])])

def _conv_next_block(dim_in, dim_out, mult=2, activation='gelu'):
    # https://arxiv.org/abs/2201.03545
    return models.Multiplex([
        tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim_in, 7, padding='same', groups=dim_in),
            GroupNorm(1),
            tf.keras.layers.Conv2D(dim_out * mult, 3, padding='same', activation=activation),
            GroupNorm(1),
            tf.keras.layers.Conv2D(dim_out, 3, padding='same'),
        ]),
        tf.keras.layers.Conv2D(dim_out, 1, padding='valid') if dim_in != dim_out else None,
    ])

def _memory_rw(params):
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

def _tokenizer_rw(tokenizer_setup, metadata_path):
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

def _text_predictor_rw(params):
    if params is None:
        return rw.KerasReadWriter('text_predictor', local_model.Predictor)

    return rw.KerasReadWriter('text_predictor', local_model.Predictor,
            params['model_dim'], params['vocab_size'])

def _text_extruder_rw(params):
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

class TextEmbed(tf.keras.Model):
    """
    TextEmbed accepts tokens and encodes into latent memory
    """
    def __init__(self, params):
        super().__init__()

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
        """
        Model call implementation
        """

        # input.shape == (batch_size, x)
        # emb.shape == (batch_size, x, model_dim)
        emb = self.embedder(inputs, training)
        # enc.shape == (batch_size, x, model_dim)
        enc = self.encoder(emb, berts.create_padding_mask(inputs), training)
        return enc

class MemoryTextProcessor(rw.StructuredModel):
    """
    MemoryTextProcessor encoding latent space accepting texts.
    """
    def __init__(self, model_path, params):
        super().__init__(model_path, {
            'text_embedder': rw.KerasReadWriter('text_embedder', TextEmbed, params),
            'memory': _memory_rw(params),
            'text_extruder': _text_extruder_rw(params),
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

class MemoryMimic(tf.keras.Model):
    """
    MemoryMimic mimics MemoryImageProcessor
    """
    def __init__(self, params):
        super().__init__()

        self.params = params

        model_dim = params['model_dim']
        self.mem = _attention(model_dim, model_dim, params)

    def call(self, inputs):
        """
        Model call implementation
        """

        # inputs.shape == (batch_size, h_dim, w_dim, model_dim)
        # mem.shape == (batch_size, h_dim, w_dim, model_dim)
        mem = self.mem(inputs)
        return mem

class ImageEmbed(tf.keras.Model):
    """
    ImageEmbed takes UnetDown latent image into latent memory
    """
    def __init__(self, params):
        super().__init__()

        self.params = params

        model_dim = params['model_dim']
        self.emb = tf.keras.Sequential([
            _attention(model_dim, model_dim, params),
            tf.keras.layers.Reshape([-1, model_dim]),
        ])

    def call(self, inputs):
        """
        Model call implementation
        """

        return self.emb(inputs)

class ImageExtrude(tf.keras.Model):
    """
    ImageExtrude takes latent memory to latent image for UnetUp
    """
    def __init__(self, params):
        super().__init__()

        self.params = params

        self.model_dim = params['model_dim']
        self.perceiver = berts.Perceiver(1, params)
        self.extrude = _attention(self.model_dim, self.model_dim, params)

    def call(self, inputs, latent, orig_shape, training=False):
        """
        Model call implementation
        """

        batch_size = orig_shape[0]
        h_dim = orig_shape[1]
        w_dim = orig_shape[2]

        out = self.perceiver(inputs, latent, training)
        out = tf.reshape(out, [batch_size, h_dim, w_dim, self.model_dim])
        out = self.extrude(out)
        return out

class MemoryImageProcessor(rw.StructuredModel):
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

class TextBert(rw.StructuredModel):
    """
    TextBert handles text transformation for <mask> token prediction and distance prediction for
    sentences separated by <sep>
    """
    def __init__(self, model_path, optimizer, params, tokenizer_setup):
        super().__init__(model_path, {
            'text_metadata': rw.FileReadWriter('text_metadata',
                lambda metadata: metadata.update({
                    'optimizer_iter': int(optimizer.iterations.numpy()), # update iterator value
                }) if optimizer is not None else None),
            'tokenizer': _tokenizer_rw(tokenizer_setup, os.path.join(model_path, 'text_metadata')),
            'memory_processor': rw.SModelReadWriter('memory_processor',
                MemoryTextProcessor, params),
            'text_predictor': _text_predictor_rw(params),
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
            opt_iter = int(self.metadata.get('optimizer_iter', 0))
            optimizer.iterations.assign(opt_iter)

    def tokenize(self, sentence):
        """
        tokenize text string use pretrainer tokenizer
        """

        out = self.tokenizer.tokenize(sentence)
        if not isinstance(out, tf.Tensor):
            out = out.to_tensor()
        return out

    def predict(self, tokens, training=False):
        """
        predict deduce the tokens replaced by <mask> and determine the distance between sentences
        separated by <sep>
        """

        assert isinstance(tokens, tf.Tensor)
        return self.call(tokens, training)

    def sentence_distance(self, distance, max_distance):
        """
        sentence_distance determines the token class mapped by the distance
        """

        return (distance * local_model.DISTANCE_CLASS) / max_distance

    def call(self, inputs, training=False):
        """
        Model call implementation
        """

        enc = self.processor(inputs, training=training)
        pred, debug_info = self.predictor(enc)
        debug_info.update({'prediction.prediction': pred})
        return (pred, debug_info)

    def get_config(self):
        """
        Module get_config implementation
        """

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
        """
        Module from_config implementation
        """

        return cls(**config)

class ImageBertPreparer(rw.StructuredModel):
    """
    ImageBert uses unet transformation techniques to encode images into latent memory and back to
    predict image noise
    """
    def __init__(
        self,
        model_path,
        nchannels,
        optimizer,
        params,
    ):
        super().__init__(model_path, {
            'image_metadata': rw.FileReadWriter('image_metadata',
                lambda metadata: metadata.update({
                    'optimizer_iter': int(optimizer.iterations.numpy()),
                }) if optimizer is not None else None),
            'image_unetd': rw.KerasReadWriter('image_unetd', diffs.MinUnetDown, nchannels, params),
            'memory_mimic': rw.KerasReadWriter('memory_mimic', MemoryMimic, params),
            'image_unetu': rw.KerasReadWriter('image_unetu', diffs.MinUnetUp, nchannels, params),
        })

        self.model_path = model_path
        self.nchannels = nchannels
        self.optimizer = optimizer
        self.params = params

        self.metadata = self.elems['image_metadata']
        self.unetd = self.elems['image_unetd']
        self.mem = self.elems['memory_mimic']
        self.unetu = self.elems['image_unetu']

        if optimizer is not None:
            optimizer.iterations.assign(int(self.metadata.get('optimizer_iter', 0)))

    def predict(self, inputs, training=False):
        """
        predict the noise of noisy inputs
        """

        return self.call(inputs, training)

    def call(self, inputs, training=False):
        """
        SaveModule call implementation
        """

        enc, hids = self.unetd(inputs)
        mem = self.mem(enc)

        hiddens = tf.TensorArray(
                tf.float32,
                size=len(hids),
                infer_shape=False,
                clear_after_read=False,
                element_shape=[None, None, None, None])
        for i, hid in enumerate(hids):
            hiddens = hiddens.write(i, hid)
        dec = self.unetu(mem, hiddens)
        return dec

class ImageBert(rw.StructuredModel):
    """
    ImageBert uses unet transformation techniques to encode images into latent memory and back to
    predict image noise
    """
    def __init__(
        self,
        model_path,
        nchannels,
        optimizer,
        params,
    ):
        super().__init__(model_path, {
            'image_metadata': rw.FileReadWriter('image_metadata',
                lambda metadata: metadata.update({
                    'optimizer_iter': int(optimizer.iterations.numpy()),
                }) if optimizer is not None else None),
            'image_unetd': rw.KerasReadWriter('image_unetd', diffs.MinUnetDown, nchannels, params),
            'memory_processor': rw.SModelReadWriter('memory_processor',
                MemoryImageProcessor, params),
            'image_unetu': rw.KerasReadWriter('image_unetu', diffs.MinUnetUp, nchannels, params),
        })

        self.model_path = model_path
        self.nchannels = nchannels
        self.optimizer = optimizer
        self.params = params

        self.metadata = self.elems['image_metadata']
        self.unetd = self.elems['image_unetd']
        self.mem = self.elems['memory_processor']
        self.unetu = self.elems['image_unetu']

        if optimizer is not None:
            optimizer.iterations.assign(int(self.metadata.get('optimizer_iter', 0)))

    def predict(self, inputs, training=False):
        """
        predict the noise of noisy inputs
        """

        return self.call(inputs, training)

    def call(self, inputs, training=False):
        """
        SaveModule call implementation
        """

        enc, hids = self.unetd(inputs)
        mem = self.mem(enc, training)

        hiddens = tf.TensorArray(
                tf.float32,
                size=len(hids),
                infer_shape=False,
                clear_after_read=False,
                element_shape=[None, None, None, None])
        dec = self.unetu(mem, hiddens)
        return dec
