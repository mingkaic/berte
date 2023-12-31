#!/usr/bin/env python3
"""
This module includes berte v3 specific models
"""

import os
import json
import shutil

import sentencepiece as sp
import tensorflow as tf
import tensorflow_text as text

import export.model as model

InitParamBuilder = model.InitParamBuilder

def best_effort_load(fpath):
    """
    load model from path if it exists otherwise None
    """
    if os.path.exists(fpath):
        return tf.keras.models.load_model(fpath)
    return None

class BerteMetadata:
    """ model just for saving metadata """
    @staticmethod
    def load(model_dir, ctx_key, optimizer):
        """ load BerteMetadata from model directory """
        with open(model_dir, 'r') as file:
            revived_obj = json.loads(file.read())

        existing_iter_map = dict()
        optimizer_iter = revived_obj['optimizer_iter']
        if isinstance(optimizer_iter, dict):
            existing_iter_map = optimizer_iter
            optimizer_iter = optimizer_iter.get(ctx_key, 0)
        elif ctx_key == 'nsp':
            optimizer_iter = 0
        optimizer.iterations.assign(optimizer_iter)
        return BerteMetadata("",
            ctx_key=ctx_key,
            optimizer_iter=optimizer.iterations,
            existing_iter_map=existing_iter_map,
            args={
                'sep': revived_obj['sep'],
                'mask': revived_obj['mask'],
                'unk': revived_obj['unk'],
                'bos': revived_obj['bos'],
                'eos': revived_obj['eos'],
                'pad': revived_obj['pad'],
            })

    def __init__(self, tokenizer_filename, ctx_key, optimizer_iter,
            existing_iter_map=dict(),
            args=None):

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
        self._ctx_key = ctx_key
        self._optimizer_iter = optimizer_iter
        self._existing_iter_map = existing_iter_map

    def optimizer_iteration(self):
        """ return optimizer iteration tf variable """
        return self._optimizer_iter

    def pad(self):
        """ return pad token value """
        return self._pad

    def mask(self):
        """ return mask token value """
        return self._mask

    def sep(self):
        """ return sep token value """
        return self._sep

    def save(self, model_dir):
        """ save metadata into model_dir path """
        self._existing_iter_map[self._ctx_key] = int(self._optimizer_iter.numpy())
        json_obj = {
            'sep': int(self._sep),
            'mask': int(self._mask),
            'unk': int(self._unk),
            'bos': int(self._bos),
            'eos': int(self._eos),
            'pad': int(self._pad),
            'optimizer_iter': self._existing_iter_map,
        }
        with open(model_dir, 'w') as file:
            json.dump(json_obj, file)

class _commonPretrainer(tf.Module):
    MODEL_ELEMS = {
        'tokenizer',
        'metadata',
        'preprocessor',
        'mask_pred',
        'ns_pred',
    }

    def __init__(self, model_path, elem_cbs):
        super().__init__()

        self.model_path = model_path
        self.elems = {elem: elem_cbs[elem][0](os.path.join(model_path, elem)) for elem in elem_cbs}
        self.savers = {elem: elem_cbs[elem][1] for elem in elem_cbs}

    def _copy_irrelevant(self, dest_path):
        for elem in _commonPretrainer.MODEL_ELEMS.difference(set(self.savers.keys())):
            source = os.path.join(self.model_path, elem)
            dest = os.path.join(dest_path, elem)
            if os.path.exists(source) and source != dest:
                # copy source to dest
                shutil.copytree(source, dest)

    def save(self, model_path):
        """ save models under model_path """
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        for elem in self.savers:
            self.savers[elem](self.elems[elem], os.path.join(model_path, elem))

        # copy over non-relevant elements
        self._copy_irrelevant(model_path)

def _load_sentencepiece(fpath, tokenizer_setup):
    with open(fpath+'.model', 'rb') as file:
        return text.SentencepieceTokenizer(
                model=file.read(),
                out_type=tf.int32,
                add_bos=tokenizer_setup["add_bos"],
                add_eos=tokenizer_setup["add_eos"])
    return None

def _load_metadata(fpath, model_path, ctx_key, optimizer):
    if os.path.exists(fpath):
        return BerteMetadata.load(fpath, ctx_key, optimizer)
    return BerteMetadata(os.path.join(model_path, 'tokenizer'), ctx_key, optimizer.iterations)

def _load_keras(fpath, kmodel, params):
    if os.path.exists(fpath):
        return tf.keras.models.load_model(fpath)
    return kmodel(params)

def _save_model(model_inst, fpath):
    model_inst.save(fpath)

def _copy_tokenizer(src, dst):
    shutil.copyfile(src+'.vocab', dst+'.vocab')
    shutil.copyfile(src+'.model', dst+'.model')

class PretrainerMLM(_commonPretrainer):
    """
    Collection of models used in pretraining using MLM
    """
    def __init__(self, model_path, optimizer, params, tokenizer_setup):
        super().__init__(model_path, {
            'tokenizer': (
                lambda fpath: _load_sentencepiece(fpath, tokenizer_setup),
                lambda _, dst: _copy_tokenizer(os.path.join(model_path, 'tokenizer'), dst)),
            'metadata': (
                lambda fpath: _load_metadata(fpath, model_path, 'mlm', optimizer), _save_model),
            'preprocessor': (
                lambda fpath: _load_keras(fpath, model.Preprocessor, params), _save_model),
            'mask_pred': (
                lambda fpath: _load_keras(fpath, model.Predictor, params), _save_model),
        })
        self.metadata = self.elems['metadata']
        self.tokenizer = self.elems['tokenizer']
        self.preprocessor = self.elems['preprocessor']
        self.mask_predictor = self.elems['mask_pred']

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

class NSPredictor(tf.keras.Model):
    """
    NSPredictor are dependent on vocab.
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.latent_dim = params['latent_dim']
        self.model_dim = params['model_dim']

        self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.vocab_pred = tf.keras.layers.Dense(1, use_bias=False)
        self.latent_pred = tf.keras.layers.Dense(1, use_bias=False, activation='sigmoid')

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
        # shape == (batch_size, latent_dim)
        vocab_p = tf.reshape(self.vocab_pred(norm_inp), [-1, self.latent_dim])
        # shape == (batch_size)
        result = tf.reshape(self.latent_pred(vocab_p), [-1])
        debug_info = {
            'nspredictor.norm_inp': norm_inp,
            'nspredictor.vocab_p': vocab_p,
            'nspredictor.result': result,
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

class PretrainerNSP(_commonPretrainer):
    """
    Collection of models used in pretraining using NSP.
    """
    def __init__(self, model_path, optimizer, params, tokenizer_setup):
        super().__init__(model_path, {
            'tokenizer': (
                lambda fpath: _load_sentencepiece(fpath, tokenizer_setup),
                lambda _, dst: _copy_tokenizer(os.path.join(model_path, 'tokenizer'), dst)),
            'metadata': (
                lambda fpath: _load_metadata(fpath, model_path, 'nsp', optimizer), _save_model),
            'preprocessor': (
                lambda fpath: _load_keras(fpath, model.Preprocessor, params), _save_model),
            'ns_pred': (lambda fpath: _load_keras(fpath, NSPredictor, params), _save_model),
        })

        self.metadata = self.elems['metadata']
        self.tokenizer = self.elems['tokenizer']
        self.preprocessor = self.elems['preprocessor']
        self.ns_predictor = self.elems['ns_pred']

    def tokenize(self, sentence):
        """ use pretrainer tokenizer """
        return self.tokenizer.tokenize(sentence).to_tensor()

    def ns_prediction(self, tokens, training=False):
        """ deduce how far the sentences separated by <sep> are """
        assert isinstance(tokens, tf.Tensor)

        _, latent = self.preprocessor(tokens, training=training)
        pred, debug_info = self.ns_predictor(latent)
        debug_info.update({
            'ns_prediction.preprocessor_latent': latent,
            'ns_prediction.prediction': pred,
        })
        return (pred, debug_info)
