#!/usr/bin/env python3
"""
This module includes berte v1 specific models
"""

import os
import functools

import sentencepiece as sp
import tensorflow as tf

import common.berts as berts

INIT_PARAMS = [
    "model_dim",
    "latent_dim",
    "num_heads",
    "max_pe",
    "dff",
    "dropout_rate",
    "num_isolation_layers",
    "num_perm_layers",
    "num_latent_layers",
    "dataset_width",
    "vocab_size",
]

class InitParamBuilder:
    """ build init parameters commonly used against the entire model """
    def __init__(self):
        self.param = dict()
        self.param["use_bias"] = False

        for key in INIT_PARAMS:
            setattr(self.__class__, key, functools.partial(self.add_param, key=key))

    def add_param(self, value, key):
        """ add custom param """
        self.param[key] = value
        return self

    def build(self):
        """ return built params """
        return self.param

def preprocessor_call_param(inp, training=False):
    """ encapsulate params """
    return {
        "inp":      inp,
        "training": training,
    }

def perceiver_call_param(enc, latent, training=False):
    """ encapsulate params """
    return {
        "enc":      enc,
        "latent":   latent,
        "training": training,
    }

class Preprocessor(tf.keras.Model):
    """
    Preprocessor are tied to vocab.
    """
    def __init__(self, params):
        super().__init__()
        num_layers = params['num_isolation_layers']
        model_dim = params['model_dim']
        latent_dim = params['latent_dim']
        num_heads = params['num_heads']
        max_pe = params['max_pe']
        dff = params['dff']
        vocab_size = params['vocab_size']
        dropout_rate = params['dropout_rate']

        self.params = params
        self.embedder = berts.InputEmbed(model_dim,
            berts.input_embed_init_builder().maximum_position_encoding(max_pe).\
            vocab_size(vocab_size).dropout_rate(dropout_rate).build())
        self.iso_enc_encoder = berts.MaskedEncoder(num_layers,
            berts.encoder_init_builder().model_dim(model_dim).num_heads(num_heads).\
            dff(dff).dropout_rate(dropout_rate).use_bias(True).build())
        self.iso_lat_encoder = berts.MaskedEncoder(num_layers,
            berts.encoder_init_builder().model_dim(model_dim).num_heads(num_heads).\
            dff(dff).dropout_rate(dropout_rate).use_bias(True).build())

    def call(self, inputs, training=False):
        """
        Model call implementation
        """
        # inc.shape == (batch_size, ?)
        # emb.shape == (batch_size, ?, model_dim)
        emb = self.embedder(inputs, training)
        # enc.shape == (batch_size, ?, model_dim)
        enc = self.iso_enc_encoder(emb, berts.create_padding_mask(inputs), training)
        # enc_lat.shape == (batch_size, ?, latent_dim)
        enc_lat = self.iso_lat_encoder(emb, berts.create_padding_mask(inputs), training)
        # lat.shape == (batch_size, latent_dim, model_dim)
        lat = tf.matmul(enc_lat, enc, transpose_a=True)

        return enc, lat

class Perceiver(tf.keras.Model):
    """
    Perceiver are independent of vocab.
    """
    def __init__(self, params):
        super().__init__()
        num_layers = params['num_perm_layers']

        self.params = params
        self.perm_perceiver = berts.Perceiver(num_layers, params)

    def call(self, inputs, latent, training=False):
        """
        Model call implementation
        """
        # enc.shape == (batch_size, ?, model_dim)
        # latent.shape == (batch_size, latent_dim, model_dim)
        return self.perm_perceiver(inputs, latent, training)

class Latenter(tf.keras.Model):
    """
    Latenter is basically an encoder.
    """
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.latenter = berts.Encoder(params['num_latent_layers'],
            berts.encoder_init_builder().model_dim(params['model_dim']).\
            num_heads(params['num_heads']).dff(params['dff']).\
            dropout_rate(params['dropout_rate']).use_bias(False).build())

    def call(self, inputs, training=False):
        """
        Model call implementation
        """
        return self.latenter(inputs, training)

class Predictor(tf.keras.Model):
    """
    Predictor are dependent on vocab.
    """
    def __init__(self, vocab_size, prediction_window):
        super().__init__()

        self.vocab_size = vocab_size
        self.prediction_window = prediction_window
        self.out = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            # shape == (batch_size, latent_dim, vocab_size)
            tf.keras.layers.Dense(vocab_size, use_bias=False),
            # shape == (batch_size, vocab_size, latent_dim)
            tf.keras.layers.Permute((2, 1)),
            tf.keras.layers.Dense(prediction_window, use_bias=False, activation='sigmoid'),
            # shape == (batch_size, latent_dim, vocab_size)
            tf.keras.layers.Permute((2, 1)),
        ])

    def call(self, inputs, training=False):
        """
        Model call implementation
        """
        # inputs.shape == (batch_size, latent_dim, model_dim)
        # shape == (batch_size, vocab_size, window_size)
        prediction = self.out(inputs, training=training)
        return prediction / tf.reduce_sum(prediction, axis=-1, keepdims=True)

class PretrainerMLMMetadataBuilder:
    """ build mlm pretrainer module metadata """
    def __init__(self):
        self.metadata = dict()

    def tokenizer_meta(self, tokenizer_filename):
        """ set tokenizer metadata """
        tmp_sp = sp.SentencePieceProcessor()
        tmp_sp.load(tokenizer_filename)

        return self.\
            add_metadata(
                tmp_sp.piece_to_id('<sep>'), 'SEP').\
            add_metadata(
                tmp_sp.piece_to_id('<mask>'), 'MASK').\
            add_metadata(tmp_sp.unk_id(), 'UNK').\
            add_metadata(tmp_sp.bos_id(), 'BOS').\
            add_metadata(tmp_sp.eos_id(), 'EOS').\
            add_metadata(tmp_sp.pad_id(), 'PAD')

    def optimizer_iter(self, value):
        """ add optimizer init """
        return self.add_metadata(value, 'optimizer_init')

    def add_metadata(self, value, key):
        """ add custom metadata """
        self.metadata[key] = value
        return self

    def build(self):
        """ return built metadata """
        return self.metadata

PRETRAINER_MLM_ARGS = [
    "preprocessor",
    "perceiver",
    "latenter",
    "mask_pred",
]

class PretrainerMLMInitBuilder:
    """ build mlm pretrainer module init arguments """
    def __init__(self):
        self.arg = dict()
        for key in PRETRAINER_MLM_ARGS:
            setattr(self.__class__, key, functools.partial(self.add_arg, key=key))

    def add_arg(self, value, key):
        """ add custom arg """
        self.arg[key] = value
        return self

    def build(self):
        """ return built arg """
        return self.arg

class PretrainerMLM(tf.Module):
    """
    Collection of models used in pretrainining using Masked Language Modeling.
    """
    @staticmethod
    def load(tokenizer, metadata, model_path):
        args = dict([(model_key, tf.keras.models.load_model(os.path.join(model_path, model_key)))
            for model_key in ['preprocessor', 'perceiver', 'latenter', 'mask_pred']])
        return PretrainerMLM(tokenizer, None, metadata, args=args)

    def __init__(self, tokenizer, params, metadata, args=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.metadata = metadata
        if args is not None:
            self.preprocessor = args["preprocessor"]
            self.perceiver = args["perceiver"]
            self.latenter = args["latenter"]
            self.mask_predictor = args["mask_pred"]
        else: # initialize models
            self.preprocessor = Preprocessor(params)
            self.perceiver = Perceiver(params)
            self.latenter = Latenter(params)
            self.mask_predictor = Predictor(params["vocab_size"],
                                            params["dataset_width"])

    def save(self, model_path):
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.preprocessor.save(os.path.join(model_path, 'preprocessor'))
        self.perceiver.save(os.path.join(model_path, 'perceiver'))
        self.latenter.save(os.path.join(model_path, 'latenter'))
        self.mask_predictor.save(os.path.join(model_path, 'mask_pred'))

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        return self.tokenizer.tokenize(sentence).to_tensor()

    def latent_prediction(self, tokens, training=False):
        """ deduce the encoding and latent starting with random latent parameter """
        assert isinstance(tokens, tf.Tensor)

        enc, latent = self.preprocessor(tokens, training=training)
        enc2 = self.perceiver(enc, latent, training)
        latent_pred = self.latenter(enc2, training)
        debug_info = {
            'latent_prediction.preprocessor_enc': enc,
            'latent_prediction.preprocessor_latent': latent,
            'latent_prediction.perceiver_enc': enc2,
            'latent_prediction.predicted_latent': latent_pred,
        }
        return (latent_pred, debug_info)

    def mask_prediction(self, tokens, latent_pred=None, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        enc, latent = self.preprocessor(tokens, training=training)
        if latent_pred is not None:
            latent = latent_pred
        enc2 = self.perceiver(enc, latent, training=training)
        pred = self.mask_predictor(enc2)
        debug_info = {
            'mask_prediction.preprocessor_enc': enc,
            'mask_prediction.preprocessor_latent': latent,
            'mask_prediction.perceiver_enc': enc2,
            'mask_prediction.prediction': pred,
        }
        return (pred, debug_info)

    def contexted_trainable_variables(self):
        """ return trainable variables with modules used to extract context """
        return self.uncontexted_trainable_variables()+\
             self.latenter.trainable_variables

    def uncontexted_trainable_variables(self):
        """ return trainable variables for mask prediction """
        return self.preprocessor.trainable_variables+\
             self.perceiver.trainable_variables+\
             self.mask_predictor.trainable_variables
