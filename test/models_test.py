#!/usr/bin/env python3
"""
This module tests models
"""
import os.path
import shutil
import unittest
import uuid

import numpy as np
import tensorflow as tf

import common.models as models

class SwiGLUModel(tf.keras.Model):
    """
    SwiGLU wrapper to test saving/loading
    """
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ])
    def call(self, inp):
        return self.layer(inp)

class MhaModel(tf.keras.Model):
    """
    MHA wrapper to test saving/loading
    """
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ])
    def call(self, arg1, arg2, arg3):
        return self.layer(arg1, arg2, arg3)

class MMhaModel(tf.keras.Model):
    """
    Masked MHA wrapper to test saving/loading
    """
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ])
    def call(self, arg1, arg2, arg3, arg4):
        return self.layer(arg1, arg2, arg3, arg4)

class ModelTest(unittest.TestCase):
    """
    Test Model module
    """
    def _test_swiglu(self, biased):
        swiglu = models.SwiGLU(5, use_bias=biased)
        swiglu_model = SwiGLUModel(swiglu)

        inp = tf.random.uniform((3, 4, 5), dtype=tf.float32)
        out = swiglu_model(inp)
        self.assertFalse(np.isnan(out.numpy()).any())

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        swiglu_model.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(inp)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def _test_unmasked_mha(self, biased):
        mha = models.UnmaskedMultiHeadAttention(15, 5, use_bias=biased)
        mha_model = MhaModel(mha)

        query = tf.random.uniform((3, 5, 15), dtype=tf.float32)
        keyval = tf.random.uniform((3, 4, 15), dtype=tf.float32)
        out = mha_model(query, keyval, keyval)
        self.assertFalse(np.isnan(out.numpy()).any())

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        mha_model.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(query, keyval, keyval)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def _test_masked_mha(self, biased):
        mha = models.MaskedMultiHeadAttention(15, 5, use_bias=biased)
        mha_model = MMhaModel(mha)

        mask = tf.random.uniform((3, 5, 5, 4), dtype=tf.float32)
        query = tf.random.uniform((3, 5, 15), dtype=tf.float32)
        keyval = tf.random.uniform((3, 4, 15), dtype=tf.float32)
        out = mha_model(query, keyval, keyval, mask)
        self.assertFalse(np.isnan(out.numpy()).any())

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        mha_model.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(query, keyval, keyval, mask)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def test_swiglu_unbiased(self):
        """
        Test swiglu with use_bias set to False
        """
        self._test_swiglu(False)

    def test_swiglu_biased(self):
        """
        Test swiglu with use_bias set to True
        """
        self._test_swiglu(True)

    def test_unmasked_mha_unbiased(self):
        """
        Test unmasked mha with use_bias set to False
        """
        self._test_unmasked_mha(False)

    def test_unmasked_mha_biased(self):
        """
        Test unmasked mha with use_bias set to True
        """
        self._test_unmasked_mha(True)

    def test_masked_mha_unbiased(self):
        """
        Test masked mha with use_bias set to False
        """
        self._test_masked_mha(False)

    def test_masked_mha_biased(self):
        """
        Test masked mha with use_bias set to True
        """
        self._test_masked_mha(True)

if __name__ == '__main__':
    unittest.main()
