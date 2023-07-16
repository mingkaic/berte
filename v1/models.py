#!/usr/bin/env python3
"""
This module includes berte v1 specific models
"""

import os
import json
import functools

import sentencepiece as sp
import tensorflow as tf
import tensorflow_text as text

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
        self.params = params
        num_layers = params['num_isolation_layers']
        model_dim = params['model_dim']
        num_heads = params['num_heads']
        max_pe = params['max_pe']
        dff = params['dff']
        vocab_size = params['vocab_size']
        dropout_rate = params['dropout_rate']

        self.embedder = berts.InputEmbed(model_dim,
            berts.input_embed_init_builder().maximum_position_encoding(max_pe).\
            vocab_size(vocab_size).dropout_rate(dropout_rate).build())
        self.iso_enc_encoder = berts.MaskedEncoder(num_layers,
            berts.encoder_init_builder().model_dim(model_dim).num_heads(num_heads).\
            dff(dff).dropout_rate(dropout_rate).use_bias(True).build())
        self.iso_lat_encoder = berts.MaskedEncoder(num_layers,
            berts.encoder_init_builder().model_dim(model_dim).num_heads(num_heads).\
            dff(dff).dropout_rate(dropout_rate).use_bias(True).build())

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
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

    def get_config(self):
        config = super().get_config()
        config.update({ 'params': self.params })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Perceiver(tf.keras.Model):
    """
    Perceiver are independent of vocab.
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        num_layers = params['num_perm_layers']

        self.perm_perceiver = berts.Perceiver(num_layers, params)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, inputs, latent, training=False):
        """
        Model call implementation
        """
        # enc.shape == (batch_size, ?, model_dim)
        # latent.shape == (batch_size, latent_dim, model_dim)
        return self.perm_perceiver(inputs, latent, training)

    def get_config(self):
        config = super().get_config()
        config.update({ 'params': self.params })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, inputs, training=False):
        """
        Model call implementation
        """
        return self.latenter(inputs, training)

    def get_config(self):
        config = super().get_config()
        config.update({ 'params': self.params })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Predictor(tf.keras.Model):
    """
    Predictor are dependent on vocab.
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.latent_dim = params['latent_dim']
        self.model_dim = params['model_dim']
        vocab_size = params['vocab_size']
        prediction_window = params["dataset_width"]

        self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.vocab_pred = tf.keras.layers.Dense(vocab_size, use_bias=False)
        self.window_pred = tf.keras.Sequential([
            tf.keras.layers.Permute((2, 1)),
            tf.keras.layers.Dense(prediction_window, use_bias=False, activation='sigmoid'),
            tf.keras.layers.Permute((2, 1)),
        ])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, inputs, training=False):
        """
        Model call implementation
        """
        # inputs.shape == (batch_size, latent_dim, model_dim)
        inputs = tf.ensure_shape(inputs, [None, self.latent_dim, self.model_dim])
        norm_inp = self.norm_layer(inputs, training=training)
        # shape == (batch_size, latent_dim, vocab_size)
        vocab_p = self.vocab_pred(norm_inp, training=training)
        # shape == (batch_size, window_size, vocab_size)
        prediction = self.window_pred(vocab_p, training=training)

        normpred = tf.sigmoid(prediction)
        npsum = tf.reduce_sum(normpred, axis=-1, keepdims=True)
        result = normpred / npsum
        debug_info = {
            'predictor.norm_inp': norm_inp,
            'predictor.vocab_p': vocab_p,
            'predictor.prediction': prediction,
            'predictor.normpred': normpred,
            'predictor.npsum': npsum,
        }
        return result, debug_info

    def get_config(self):
        config = super().get_config()
        config.update({
            'params': self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class BerteMetadata:
    """ model just for saving metadata """
    @staticmethod
    def load(model_dir, optimizer):
        """ load BerteMetadata from model directory """
        with open(model_dir, 'r') as file:
            revived_obj = json.loads(file.read())
        optimizer.iterations.assign(revived_obj['optimizer_iter'])
        return BerteMetadata("",
            optimizer_iter=optimizer.iterations,
            args={
                'sep': revived_obj['sep'],
                'mask': revived_obj['mask'],
                'unk': revived_obj['unk'],
                'bos': revived_obj['bos'],
                'eos': revived_obj['eos'],
                'pad': revived_obj['pad'],
            })

    def __init__(self, tokenizer_filename, optimizer_iter=0, args=None):
        if args is None:
            tmp_sp = sp.SentencePieceProcessor()
            tmp_sp.load(tokenizer_filename)
            self._sep = tmp_sp.piece_to_id('<sep>')
            self._mask = tmp_sp.piece_to_id('<mask>')
            self._unk = tmp_sp.unk_id()
            self._bos = tmp_sp.bos_id()
            self._eos = tmp_sp.eos_id()
            self._pad = tmp_sp.pad_id()
        else:
            self._sep = args['sep']
            self._mask = args['mask']
            self._unk = args['unk']
            self._bos = args['bos']
            self._eos = args['eos']
            self._pad = args['pad']
        self._optimizer_iter = tf.Variable(optimizer_iter, trainable=False)

    def optimizer_iteration(self):
        return self._optimizer_iter

    def pad(self):
        return self._pad

    def save(self, model_dir):
        json_obj = {
            'sep': int(self._sep),
            'mask': int(self._mask),
            'unk': int(self._unk),
            'bos': int(self._bos),
            'eos': int(self._eos),
            'pad': int(self._pad),
            'optimizer_iter': int(self._optimizer_iter.numpy()),
        }
        with open(model_dir, 'w') as file:
            json.dump(json_obj, file)

class PretrainerMLM(tf.Module):
    """
    Collection of models used in pretrainining using Masked Language Modeling.
    """
    @staticmethod
    def load(model_path, optimizer, tokenizer_filename, tokenizer_setup):
        """ load PretrainerMLM from tokenizer_filename and parameters under model_path """
        with open(tokenizer_filename, 'rb') as file:
            tokenizer = text.SentencepieceTokenizer(model=file.read(),
                                                    out_type=tf.int32,
                                                    add_bos=tokenizer_setup["add_bos"],
                                                    add_eos=tokenizer_setup["add_eos"])

        metadata_path = os.path.join(model_path, 'metadata')
        if os.path.exists(metadata_path):
            metadata = BerteMetadata.load(metadata_path, optimizer)
        else:
            metadata = BerteMetadata(tokenizer_filename, optimizer.iterations)

        args = {model_key:tf.keras.models.load_model(os.path.join(model_path, model_key))
            for model_key in ['preprocessor', 'perceiver', 'latenter', 'mask_pred']}
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
            self.mask_predictor = Predictor(params)

    def save(self, model_path):
        """ save models under model_path """
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.metadata.save(os.path.join(model_path, 'metadata'))
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
        pred, debug_info = self.mask_predictor(enc2)
        debug_info.update({
            'mask_prediction.preprocessor_enc': enc,
            'mask_prediction.preprocessor_latent': latent,
            'mask_prediction.perceiver_enc': enc2,
            'mask_prediction.prediction': pred,
        })
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
