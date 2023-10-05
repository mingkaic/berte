#!/usr/bin/env python3
"""
This module includes berte v2 specific models
"""

import os

import tensorflow as tf
import tensorflow_text as text

import common.berts as berts

from intake.models import *

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
