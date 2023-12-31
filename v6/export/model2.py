#!/usr/bin/env python3
"""
This module includes berte specific models
"""

import os

import tensorflow as tf

import export.model as model

class Pretrainer(model._commonPretrainer):
    """
    Collection of models used in pretraining using NSP.
    """
    def __init__(self, model_path, optimizer, params, tokenizer_setup):
        super().__init__(model_path, {
            'tokenizer': (lambda fpath: model._load_sentencepiece(fpath, tokenizer_setup),
                lambda _, dst: model._copy_tokenizer(os.path.join(model_path, 'tokenizer'), dst)),
            'metadata': (lambda fpath: model._load_metadata(fpath, model_path, 'mlm', optimizer), model._save_model),
            'processor': (lambda fpath: model._load_keras(fpath, model.MemoryProcessor, params), model._save_model),
            'mask_pred': (lambda fpath: model._load_keras(fpath, model.Predictor,
                params['model_dim'], params['vocab_size']), model._save_model),
            'ns_pred': (lambda fpath: model._load_keras(fpath, model.Predictor,
                params['model_dim'], 1), model._save_model),
        })

        self.metadata = self.elems['metadata']
        self.tokenizer = self.elems['tokenizer']
        self.processor = self.elems['processor']
        self.mask_predictor = self.elems['mask_pred']
        self.ns_predictor = self.elems['ns_pred']

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        return self.tokenizer.tokenize(sentence).to_tensor()

    def mask_prediction(self, tokens, training=False):
        """ deduce how far the sentences separated by <sep> are """
        assert isinstance(tokens, tf.Tensor)

        enc = self.processor(tokens, training=training)
        mask_pred, debug_info = self.mask_predictor(enc)
        debug_info.update({'prediction.mask_prediction': mask_pred})
        return (mask_pred, debug_info)

    def ns_prediction(self, tokens, training=False):
        """ deduce how far the sentences separated by <sep> are """
        assert isinstance(tokens, tf.Tensor)

        enc = self.processor(tokens, training=training)
        ns_pred, debug_info = self.ns_predictor(enc)
        debug_info.update({'prediction.ns_prediction': ns_pred})
        return (ns_pred, debug_info)
