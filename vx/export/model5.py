#!/usr/bin/env python3
"""
This module includes berte v4 specific models
"""

import os

import tensorflow as tf

from export.model import Predictor, Preprocessor
import export.model3 as model
import common.berts as berts

def perceiver_call_param(enc, latent, training=False):
    """ encapsulate params """
    return {
        "enc":      enc,
        "latent":   latent,
        "training": training,
    }

class Perceiver(tf.keras.Model):
    """
    Perceiver are independent of vocab.
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.perm_perceiver = berts.Perceiver(params['num_perm_layers'], params)

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

    def multi_call(self, inputs, latent, training=False, *args):
        """
        calls with multiple latents
        """
        return self.perm_perceiver(inputs, latent, *args, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({ 'params': self.params })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# forward export this
CommonPretrainer = model._commonPretrainer

load_keras = model._load_keras

save_model = model._save_model

class PretrainerPreprocessor(CommonPretrainer):
    """
    Collection of models used in pretraining.
    """
    def __init__(self, model_path, optimizer, params, tokenizer_setup,
            training_type='mlm'):
        super().__init__(model_path, {
            'tokenizer': (
                lambda fpath: model._load_sentencepiece(fpath, tokenizer_setup),
                lambda _, dst: model._copy_tokenizer(os.path.join(model_path, 'tokenizer'), dst)),
            'metadata': (lambda fpath: model._load_metadata(
                fpath, model_path, training_type, optimizer), save_model),
            'preprocessor': (
                lambda fpath: load_keras(fpath, Preprocessor, params), save_model),
        })

        self.metadata = self.elems['metadata']
        self.tokenizer = self.elems['tokenizer']
        self.preprocessor = self.elems['preprocessor']

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        return self.tokenizer.tokenize(sentence).to_tensor()

    def preprocess(self, tokens, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        enc, latent = self.preprocessor(tokens, training=training)
        debug_info = {'preprocess.enc': enc, 'preprocess.latent': latent}
        return enc, latent, debug_info

class ExtendedPretrainerMLM(CommonPretrainer):
    """
    Collection of extended models used in pretraining using Masked Language Modeling.
    """
    def __init__(self, model_path, params):
        super().__init__(model_path, {
            'perceiver': (lambda fpath: load_keras(fpath, Perceiver, params), save_model),
            'mask_pred': (lambda fpath: load_keras(fpath, Predictor, params), save_model),
        })

        self.perceiver = self.elems['perceiver']
        self.predictor = self.elems['mask_pred']

    def ml_prediction(self, enc, latent, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(enc, tf.Tensor)
        assert isinstance(latent, tf.Tensor)

        perception = self.perceiver(enc, latent, training=training)
        pred, debug_info = self.predictor(perception)
        debug_info.update({'ml_prediction.perceiver': perception, 'ml_prediction.pred': pred})
        return (pred, debug_info)

    def multi_latent_ml_prediction(self, enc, latent, training=False, *args):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(enc, tf.Tensor)
        assert isinstance(latent, tf.Tensor)

        print('>>>', flush=True)
        print(dir(self.perceiver), flush=True)
        perception = self.perceiver.multi_call(enc, latent,
                training=training, *args)
        pred, debug_info = self.predictor(perception)
        debug_info.update({'ml_prediction.perceiver': perception, 'ml_prediction.pred': pred})
        return (pred, debug_info)

class ExtendedPretrainerNSP(CommonPretrainer):
    """
    Collection of extended models used in pretraining using Next Sentence Prediction.
    """
    def __init__(self, model_path, params):
        super().__init__(model_path, {
            'perceiver': (lambda fpath: load_keras(fpath, Perceiver, params), save_model),
            'ns_pred': (lambda fpath: load_keras(fpath, model.NSPredictor, params), save_model),
        })

        self.perceiver = self.elems['perceiver']
        self.predictor = self.elems['ns_pred']

    def ns_prediction(self, enc, latent, training=False):
        """ deduce how far the sentences separated by <sep> are """
        assert isinstance(enc, tf.Tensor)
        assert isinstance(latent, tf.Tensor)

        perception = self.perceiver(enc, latent, training=training)
        pred, debug_info = self.predictor(perception)
        debug_info.update({'ns_prediction.perceiver': perception, 'ns_prediction.pred': pred})
        return (pred, debug_info)
