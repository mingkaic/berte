#!/usr/bin/env python3
"""
This module includes berte v4 specific models
"""

import os

import tensorflow as tf
import tensorflow_text as text

import common.berts as berts

from intake.model import Predictor
from intake.model3 import Pretrainer

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

class PerceiverExtendedPretrainerMLM(tf.Module):
    """
    Collection of models used in pretrainining using Masked Language Modeling.
    """
    @staticmethod
    def load(model_path, optimizer, tokenizer_filename, tokenizer_setup):
        """ load PerceiverExtendedPretrainerMLM from tokenizer_filename and parameters under model_path """

        preprocessor = Pretrainer.load(os.path.join(model_path, 'preprocessor'),
                optimizer, tokenizer_filename, tokenizer_setup)
        args = {model_key:tf.keras.models.load_model(os.path.join(model_path, model_key))
            for model_key in ['perceiver', 'mask_pred', 'ns_width']}

        return PerceiverExtendedPretrainerMLM(None, preprocessor, args)

    def __init__(self, params, preprocessor, args=None):
        super().__init__()

        self.preprocessor = preprocessor
        if args is not None:
            self.perceiver = args['perceiver']
            self.mask_predictor = args['mask_pred']
            self.ns_predictor = args['ns_pred']
        else:
            self.perceiver = Perceiver(params)
            self.mask_predictor = Predictor(params)
            params['data_width'] = 1
            self.ns_predictor = Predictor(params)

    def save(self, model_path):
        """ save models under model_path """
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.preprocessor.save(os.path.join(model_path, 'preprocessor'))
        self.perceiver.save(os.path.join(model_path, 'perceiver'))
        self.mask_predictor.save(os.path.join(model_path, 'mask_pred'))
        self.ns_predictor.save(os.path.join(model_path, 'ns_pred'))

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        return self.preprocessor.tokenize(sentence)

    def mask_prediction(self, tokens, latent_pred=None, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        enc, latent = self.preprocessor.preprocessor(tokens, training=training)
        if latent_pred is not None:
            latent = latent_pred
        enc2 = self.perceiver(enc, latent, training=training)
        pred, debug_info = self.mask_predictor(enc2)
        debug_info.update({
            'mask_prediction.preprocessor_enc': enc,
            'mask_prediction.latent': latent,
            'mask_prediction.perceiver_enc': enc2,
            'mask_prediction.prediction': pred,
        })
        return (pred, debug_info)

    def ns_prediction(self, tokens, latent_pred=None, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        enc, latent = self.preprocessor.preprocessor(tokens, training=training)
        if latent_pred is not None:
            latent = latent_pred
        enc2 = self.perceiver(enc, latent, training=training)
        pred, debug_info = self.ns_predictor(enc2)
        debug_info.update({
            'ns_prediction.preprocessor_enc': enc,
            'ns_prediction.latent': latent,
            'ns_prediction.perceiver_enc': enc2,
            'ns_prediction.prediction': pred,
        })
        return (pred, debug_info)

    def pp_trainable_variables(self):
        """ return trainable variables with preprocessor """
        return self.preprocessor.trainable_variables

    def mlm_trainable_variables(self):
        """ return trainable variables with perceiver and mlm predictor """
        return self.perceiver.trainable_variables+\
                self.mask_predictor.trainable_variables

    def ns_trainable_variables(self):
        """ return trainable variables with perceiver and ns predictor """
        return self.perceiver.trainable_variables+\
                self.ns_predictor.trainable_variables
