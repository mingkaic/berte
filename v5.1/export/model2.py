#!/usr/bin/env python3
"""
This module includes berte specific models
"""

import os
import json

import sentencepiece as sp
import tensorflow as tf

import export.model as model

class BerteMetadata:
    """ model just for saving metadata """
    @staticmethod
    def load(model_dir, optimizer):
        """ load BerteMetadata from model directory """
        with open(model_dir, 'r') as file:
            revived_obj = json.loads(file.read())

        optimizer_iter = revived_obj.get('optimizer_iter', 0)
        optimizer.iterations.assign(optimizer_iter)
        return BerteMetadata("",
            optimizer_iter=optimizer.iterations,
            args={
                'mask': revived_obj['mask'],
                'cls': revived_obj['cls'],
                'sep': revived_obj['sep'],
                'unk': revived_obj['unk'],
                'bos': revived_obj['bos'],
                'eos': revived_obj['eos'],
                'pad': revived_obj['pad'],
            })

    def __init__(self, tokenizer_filename, optimizer_iter, args=None):

        if args is None:
            tmp_sp = sp.SentencePieceProcessor()
            tmp_sp.load(tokenizer_filename)
            self._mask = tmp_sp.piece_to_id('<mask>')
            self._cls = [tmp_sp.piece_to_id('<cls>')]+\
                [tmp_sp.piece_to_id('<cls{}>'.format(i)) for i in range(1, model.DISTANCE_CLASSES)]
            self._sep = tmp_sp.piece_to_id('<sep>')
            self._unk = tmp_sp.unk_id()
            self._bos = tmp_sp.bos_id()
            self._eos = tmp_sp.eos_id()
            self._pad = tmp_sp.pad_id()
        else:
            self._mask = args['mask']
            self._cls = args['cls']
            self._sep = args['sep']
            self._unk = args['unk']
            self._bos = args['bos']
            self._eos = args['eos']
            self._pad = args['pad']
        self._optimizer_iter = optimizer_iter

    def optimizer_iteration(self):
        """ return optimizer iteration tf variable """
        return self._optimizer_iter

    def pad(self):
        """ return pad token value """
        return self._pad

    def mask(self):
        """ return mask token value """
        return self._mask

    def cls(self):
        """ return cls token value """
        return self._cls

    def sep(self):
        """ return sep token value """
        return self._sep

    def save(self, model_dir):
        """ save metadata into model_dir path """
        json_obj = {
            'mask': int(self._mask),
            'sep': int(self._sep),
            'cls': [int(c) for c in self._cls],
            'unk': int(self._unk),
            'bos': int(self._bos),
            'eos': int(self._eos),
            'pad': int(self._pad),
            'optimizer_iter': int(self._optimizer_iter.numpy()),
        }
        with open(model_dir, 'w') as file:
            json.dump(json_obj, file)

def _load_metadata(fpath, model_path, optimizer):
    if os.path.exists(fpath):
        return BerteMetadata.load(fpath, optimizer)
    return BerteMetadata(os.path.join(model_path, 'tokenizer.model'), optimizer.iterations)

class Pretrainer(model._commonPretrainer):
    """
    Collection of models used in pretraining using MLM
    """
    def __init__(self, model_path, optimizer, params, tokenizer_setup):
        super().__init__(model_path, {
            'tokenizer': (
                lambda fpath: model._load_sentencepiece(fpath, tokenizer_setup),
                lambda _, dst: model._copy_tokenizer(os.path.join(model_path, 'tokenizer'), dst)),
            'metadata': (
                lambda fpath: _load_metadata(fpath, model_path, optimizer), model._save_model),
            'processor': (
                lambda fpath: model._load_keras(fpath, model.MemoryProcessor, params),
                model._save_model),
            'predictor': (
                lambda fpath: model._load_keras(fpath, model.Predictor,
                    params['model_dim'], params['vocab_size']),
                model._save_model),
        })
        self.metadata = self.elems['metadata']
        self.tokenizer = self.elems['tokenizer']
        self.processor = self.elems['processor']
        self.predictor = self.elems['predictor']

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        out = self.tokenizer.tokenize(sentence)
        if not isinstance(out, tf.Tensor):
            out = out.to_tensor()
        return out

    def predict(self, tokens, training=False):
        """ deduce the tokens replaced by <mask> """
        assert isinstance(tokens, tf.Tensor)

        enc = self.processor(tokens, training=training)
        pred, debug_info = self.predictor(enc)
        debug_info.update({'prediction.prediction': pred})
        return (pred, debug_info)
