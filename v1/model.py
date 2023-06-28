#!/usr/bin/env python3
"""
This module includes berte v1 specific models
"""

import functools

import sentencepiece as sp
import tensorflow as tf

import common.bert as bert

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
    @staticmethod
    def from_loaded(loaded):
        return Preprocessor(None,
                args={
                    "embedder": bert.InputEmbed.from_loaded(loaded.embedder),
                    "iso_enc_encoder": bert.Encoder.from_loaded(loaded.iso_enc_encoder),
                    "iso_lat_encoder": bert.Encoder.from_loaded(loaded.iso_lat_encoder),
                })

    def __init__(self, params, args=None):
        super().__init__()

        if args is not None:
            self.embedder = args["embedder"]
            self.iso_enc_encoder = args["iso_enc_encoder"]
            self.iso_lat_encoder = args["iso_lat_encoder"]
        else:
            num_layers = params["num_isolation_layers"]
            model_dim = params["model_dim"]
            latent_dim = params["latent_dim"]
            num_heads = params["num_heads"]
            max_pe = params["max_pe"]
            dff = params["dff"]
            vocab_size = params["vocab_size"]
            dropout_rate = params["dropout_rate"]
            self.embedder = bert.InputEmbed(
                bert.input_embed_init_builder().model_dim(model_dim).\
                maximum_position_encoding(max_pe).vocab_size(vocab_size).\
                dropout_rate(dropout_rate).build())
            self.iso_enc_encoder = bert.Encoder(num_layers,
                bert.encoder_init_builder().model_dim(model_dim).num_heads(num_heads).\
                dff(dff).dropout_rate(dropout_rate).use_bias(True).build())
            self.iso_lat_encoder = bert.Encoder(num_layers,
                bert.encoder_init_builder().model_dim(model_dim).num_heads(num_heads).\
                dff(dff).dropout_rate(dropout_rate).use_bias(True).build())

    def call(self, inputs, training=False, mask=None):
        """
        Layer call implementation
        """
        inp = inputs["inp"]
        training = inputs.get("training", training)
        # inc.shape == (batch_size, ?)
        # emb.shape == (batch_size, ?, model_dim)
        emb = self.embedder(bert.input_embed_call_builder().inp(inp).training(training).build())
        # enc.shape == (batch_size, ?, model_dim)
        enc, atn_weights = self.iso_enc_encoder(
            bert.encoder_call_builder().\
                enc(emb).mask(bert.create_padding_mask(inp)).training(training).build(),
            training=training)

        # enc_lat.shape == (batch_size, ?, latent_dim)
        enc_lat, atn_weights2 = self.iso_lat_encoder(
            bert.encoder_call_builder().\
                enc(emb).mask(bert.create_padding_mask(inp)).training(training).build(),
            training=training)
        # lat.shape == (batch_size, latent_dim, model_dim)
        lat = tf.matmul(enc_lat, enc, transpose_a=True)

        atn_weights.update(atn_weights2)
        return enc, lat, atn_weights

class Perceiver(tf.keras.Model):
    """
    Perceiver are independent of vocab.
    """
    @staticmethod
    def from_loaded(loaded):
        return Perceiver(None, perm_perceiver=bert.Perceiver.from_loaded(loaded.perm_perceiver))

    def __init__(self, params, perm_perceiver=None):
        super().__init__()

        if perm_perceiver is not None:
            self.perm_perceiver = perm_perceiver
        else:
            num_layers = params["num_perm_layers"]
            self.perm_perceiver = bert.Perceiver(num_layers, params)

    def call(self, inputs, training=False, mask=None):
        """
        Layer call implementation
        """
        enc = inputs["enc"]
        latent = inputs["latent"]
        training = inputs.get("training", training)
        # enc.shape == (batch_size, ?, model_dim)
        # latent.shape == (batch_size, latent_dim, model_dim)
        return self.perm_perceiver(bert.perceiver_call_builder().\
                enc(enc).latent(latent).mask(mask).training(training).build())

class Predictor(tf.keras.Model):
    """
    Predictor are dependent on vocab.
    """
    @staticmethod
    def from_loaded(loaded):
        return Predictor(None, None, out=loaded.out)

    def __init__(self, vocab_size, prediction_window, out=None):
        super().__init__()

        if out is not None:
            self.out = out
        else:
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

    def call(self, inputs, training=None, mask=None):
        """
        Layer call implementation
        """
        # enc.shape == (batch_size, latent_dim, model_dim)
        prediction = self.out(inputs) # shape == (batch_size, vocab_size, window_size)
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
                tf.Variable(tmp_sp.piece_to_id('<sep>'), trainable=False, name='sep'), 'SEP').\
            add_metadata(
                tf.Variable(tmp_sp.piece_to_id('<mask>'), trainable=False, name='mask'), 'MASK').\
            add_metadata(tf.Variable(tmp_sp.unk_id(), trainable=False, name='unk'), 'UNK').\
            add_metadata(tf.Variable(tmp_sp.bos_id(), trainable=False, name='bos'), 'BOS').\
            add_metadata(tf.Variable(tmp_sp.eos_id(), trainable=False, name='eos'), 'EOS').\
            add_metadata(tf.Variable(tmp_sp.pad_id(), trainable=False, name='pad'), 'PAD')

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
    def from_loaded(tokenizer, loaded):
        return PretrainerMLM(tokenizer,
                loaded.params,
                loaded.metadata,
                args={
                    "preprocessor": Preprocessor.from_loaded(loaded.preprocessor),
                    "perceiver":  Perceiver.from_loaded(loaded.perceiver),
                    "latenter":  bert.Encoder.from_loaded(loaded.latenter),
                    "mask_pred":  Predictor.from_loaded(loaded.mask_predictor),
                })

    def __init__(self, tokenizer, params, metadata, args=None):
        super().__init__()

        self.tokenizer = tokenizer
        if args is not None:
            self.preprocessor = args["preprocessor"]
            self.perceiver = args["perceiver"]
            self.latenter = args["latenter"]
            self.mask_predictor = args["mask_pred"]
        else: # initialize models
            self.preprocessor = Preprocessor(params)
            self.perceiver = Perceiver(params)
            self.latenter = bert.Encoder(params["num_latent_layers"],
                bert.encoder_init_builder().model_dim(params["model_dim"]).\
                num_heads(params["num_heads"]).dff(params["dff"]).\
                dropout_rate(params["dropout_rate"]).use_bias(False).build())
            self.mask_predictor = Predictor(params["vocab_size"],
                                            params["dataset_width"])

        # for persistence
        self.params = params
        self.metadata = metadata

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        return self.tokenizer.tokenize(sentence).to_tensor()

    def latent_prediction(self, tokens, training=False):
        """ deduce the encoding and latent starting with random latent parameter """
        assert isinstance(tokens, tf.Tensor)

        enc, latent, _ = self.preprocessor(
            preprocessor_call_param(tokens, training=training),
            training=training)
        enc2, _ = self.perceiver(
            bert.perceiver_call_builder().enc(enc).latent(latent).training(training).build(),
            training=training)
        latent_pred, _ = self.latenter(
            bert.encoder_call_builder().enc(enc2).training(training).build(),
            training=training)
        debug_info = {
            'latent_prediction.preprocessor_enc': enc,
            'latent_prediction.preprocessor_latent': latent,
            'latent_prediction.perceiver_enc': enc2,
            'latent_prediction.predicted_latent': latent_pred,
        }
        return (enc, latent_pred, debug_info)

    def mask_prediction(self, tokens, latent_pred=None, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        enc, latent, _ = self.preprocessor(
                preprocessor_call_param(tokens, training=training), training=training)
        if latent_pred is not None:
            latent = latent_pred
        enc2, _ = self.perceiver(
                perceiver_call_param(enc, latent, training=training), training=training)
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
