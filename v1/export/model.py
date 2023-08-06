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
            tf.keras.layers.Dense(prediction_window, use_bias=False),
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
        vocab_p = self.vocab_pred(norm_inp)
        # shape == (batch_size, window_size, vocab_size)
        pred = self.window_pred(vocab_p)

        result, _ = tf.linalg.normalize(pred, ord=1, axis=-1)
        # shape == (batch_size, window_size)
        debug_info = {
            'predictor.norm_inp': norm_inp,
            'predictor.vocab_p': vocab_p,
            'predictor.prediction': pred,
            'predictor.normpred': result,
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

    def __init__(self, tokenizer_filename, optimizer_iter, args=None):
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
        self._optimizer_iter = optimizer_iter

    def optimizer_iteration(self):
        return self._optimizer_iter

    def pad(self):
        return self._pad

    def mask(self):
        return self._mask

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

class SimplePretrainerMLM(tf.Module):
    """
    Collection of models used in pretrainining using Masked Language Modeling.
    """
    @staticmethod
    def load(model_path, optimizer, tokenizer_filename, tokenizer_setup):
        """ load SimplePretrainerMLM from tokenizer_filename and parameters under model_path """
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
            for model_key in ['preprocessor', 'mask_pred']}
        return SimplePretrainerMLM(tokenizer, None, metadata, args=args)

    def __init__(self, tokenizer, params, metadata, args=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.metadata = metadata
        if args is not None:
            self.preprocessor = args["preprocessor"]
            self.mask_predictor = args["mask_pred"]
        else: # initialize models
            self.preprocessor = Preprocessor(params)
            self.mask_predictor = Predictor(params)

    def save(self, model_path):
        """ save models under model_path """
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.metadata.save(os.path.join(model_path, 'metadata'))
        self.preprocessor.save(os.path.join(model_path, 'preprocessor'))
        self.mask_predictor.save(os.path.join(model_path, 'mask_pred'))

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        return self.tokenizer.tokenize(sentence).to_tensor()

    def mask_prediction(self, tokens, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        _, latent = self.preprocessor(tokens, training=training)
        pred, debug_info = self.mask_predictor(latent)
        debug_info.update({
            'mask_prediction.preprocessor_latent': latent,
            'mask_prediction.prediction': pred,
        })
        return (pred, debug_info)
