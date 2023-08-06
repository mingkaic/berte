#!/usr/bin/env python3
"""
This module includes berte v3 specific models
"""

import os

import tensorflow as tf
import tensorflow_text as text

import intake.model as model

def best_effort_load(fpath):
    if os.path.exists(fpath):
        return tf.keras.models.load_model(fpath)
    return None

class Pretrainer(tf.Module):
    """
    Collection of models used in pretraining using MLM and NSP.
    """
    @staticmethod
    def load(model_path, optimizer, tokenizer_filename, tokenizer_setup, params):
        """ load Pretrainer from tokenizer_filename and parameters under model_path """
        with open(tokenizer_filename, 'rb') as file:
            tokenizer = text.SentencepieceTokenizer(model=file.read(),
                                                    out_type=tf.int32,
                                                    add_bos=tokenizer_setup["add_bos"],
                                                    add_eos=tokenizer_setup["add_eos"])

        metadata_path = os.path.join(model_path, 'metadata')
        if os.path.exists(metadata_path):
            metadata = model.BerteMetadata.load(metadata_path, optimizer)
        else:
            metadata = model.BerteMetadata(tokenizer_filename, optimizer.iterations)

        args = {model_key:best_effort_load(os.path.join(model_path, model_key))
            for model_key in ['preprocessor', 'mask_pred', 'ns_pred']}
        return Pretrainer(tokenizer, params, metadata, args=args)

    def __init__(self, tokenizer, params, metadata, args=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.metadata = metadata
        if args is not None:
            self.preprocessor = args['preprocessor']
            self.mask_predictor = args['mask_pred']
            self.ns_predictor = args['ns_pred']

        # initialize models
        if self.preprocessor is None:
            self.preprocessor = model.Preprocessor(params)

        if self.mask_predictor is None:
            self.mask_predictor = model.Predictor(params)

        if self.ns_predictor is None:
            params['dataset_width'] = 1
            self.ns_predictor = model.Predictor(params)

    def save(self, model_path):
        """ save models under model_path """
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.metadata.save(os.path.join(model_path, 'metadata'))
        self.preprocessor.save(os.path.join(model_path, 'preprocessor'))
        self.mask_predictor.save(os.path.join(model_path, 'mask_pred'))
        self.ns_predictor.save(os.path.join(model_path, 'ns_pred'))

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

    def ns_prediction(self, tokens, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        _, latent = self.preprocessor(tokens, training=training)
        pred, debug_info = self.ns_predictor(latent)
        debug_info.update({
            'ns_prediction.preprocessor_latent': latent,
            'ns_prediction.prediction': pred,
        })
        return (pred, debug_info)

    def mlm_trainable_variables(self):
        return self.preprocessor.trainable_variables+\
                self.mask_predictor.trainable_variables

    def ns_trainable_variables(self):
        return self.preprocessor.trainable_variables+\
                self.ns_predictor.trainable_variables
