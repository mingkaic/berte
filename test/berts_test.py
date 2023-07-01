"""
This module tests berts
"""
import os.path
import shutil
import unittest
import uuid

import numpy as np
import tensorflow as tf

import common.berts as berts

class PerceiverLayerModel(tf.keras.Model):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, arg0, arg1, arg2):
        return self.layer(arg0, arg1, arg2)

class EncoderLayerModel(tf.keras.Model):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, arg0, arg1):
        return self.layer(arg0, arg1)

class MaskedEncoderLayerModel(tf.keras.Model):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, arg0, arg1, arg2):
        return self.layer(arg0, arg1, arg2)

class ModelTest(unittest.TestCase):
    def _test_perceiver_layer(self, biased):
        perceiver = berts._perceiverLayer({
            'model_dim': 16,
            'num_heads': 8,
            'dff': 32,
            'dropout_rate': 0.4,
            'use_bias': biased,
        })
        pmodel = PerceiverLayerModel(perceiver)

        inp    = tf.Variable((np.random.rand(3, 9, 16) + 10) / 5, dtype=tf.float32)
        latent = tf.Variable((np.random.rand(3, 8, 16) + 10) / 5, dtype=tf.float32)
        out = pmodel(inp, latent, False)

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        pmodel.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(inp, latent, False)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def _test_unmasked_encoder_layer(self, biased):
        encoder = berts._encoderLayer({
            'model_dim': 16,
            'num_heads': 8,
            'dff': 32,
            'dropout_rate': 0.4,
            'use_bias': biased,
        })
        emodel = EncoderLayerModel(encoder)

        inp = tf.Variable((np.random.rand(3, 9, 16) + 10) / 5, dtype=tf.float32)
        out = emodel(inp, False)

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        emodel.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(inp, False)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def _test_masked_encoder_layer(self, biased):
        encoder = berts._maskedEncoderLayer({
            'model_dim': 16,
            'num_heads': 8,
            'dff': 32,
            'dropout_rate': 0.4,
            'use_bias': biased,
        })
        emodel = MaskedEncoderLayerModel(encoder)

        inp = tf.Variable((np.random.rand(3, 9, 16) + 10) / 5, dtype=tf.float32)
        mask = tf.Variable((np.random.rand(3, 8, 9, 9) + 10) / 5, dtype=tf.float32)
        out = emodel(inp, mask, False)

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        emodel.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(inp, mask, False)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def test_perceiver_layer_unbiased(self):
        """
        Test perceiver layer with use_bias set to False
        """
        self._test_perceiver_layer(False)

    def test_perceiver_layer_biased(self):
        """
        Test perceiver layer with use_bias set to True
        """
        self._test_perceiver_layer(True)

    def test_unmasked_encoder_layer_unbiased(self):
        """
        Test encoder layer with use_bias set to False
        """
        self._test_unmasked_encoder_layer(False)

    def test_unmasked_encoder_layer_biased(self):
        """
        Test encoder layer with use_bias set to True
        """
        self._test_unmasked_encoder_layer(True)

    def test_masked_encoder_layer_unbiased(self):
        """
        Test masked encoder layer with use_bias set to False
        """
        self._test_masked_encoder_layer(False)

    def test_masked_encoder_layer_biased(self):
        """
        Test masked encoder layer with use_bias set to True
        """
        self._test_masked_encoder_layer(True)

if __name__ == '__main__':
    unittest.main()
