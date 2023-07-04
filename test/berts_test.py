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

class InputEmbedModel(tf.keras.Model):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, arg0, arg1):
        return self.layer(arg0, arg1)

class PerceiverModel(tf.keras.Model):
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

class EncoderModel(tf.keras.Model):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ])
    def call(self, arg0, arg1):
        return self.layer(arg0, arg1)

class MaskedEncoderModel(tf.keras.Model):
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
        pmodel = PerceiverModel(perceiver)

        inp = tf.random.uniform((3, 9, 16), dtype=tf.float32)
        latent = tf.random.uniform((3, 8, 16), dtype=tf.float32)
        out = pmodel(inp, latent, False)
        self.assertFalse(np.isnan(out.numpy()).any())

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
        emodel = EncoderModel(encoder)

        inp = tf.random.uniform((3, 9, 16), dtype=tf.float32)
        out = emodel(inp, False)
        self.assertFalse(np.isnan(out.numpy()).any())

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
        emodel = MaskedEncoderModel(encoder)

        inp = tf.random.uniform((3, 9, 16), dtype=tf.float32)
        mask = tf.random.uniform((3, 8, 9, 9), dtype=tf.float32)
        out = emodel(inp, mask, False)
        self.assertFalse(np.isnan(out.numpy()).any())

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        emodel.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(inp, mask, False)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def _test_perceiver(self, biased):
        perceiver = berts.Perceiver(3, {
            'model_dim': 16,
            'num_heads': 8,
            'dff': 32,
            'dropout_rate': 0.4,
            'use_bias': biased,
        })
        pmodel = PerceiverModel(perceiver)

        inp = tf.random.uniform((3, 9, 16), dtype=tf.float32)
        latent = tf.random.uniform((3, 8, 16), dtype=tf.float32)
        out = pmodel(inp, latent, False)
        self.assertFalse(np.isnan(out.numpy()).any())

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        pmodel.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(inp, latent, False)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def _test_unmasked_encoder(self, biased):
        encoder = berts.Encoder(3, {
            'model_dim': 16,
            'num_heads': 8,
            'dff': 32,
            'dropout_rate': 0.4,
            'use_bias': biased,
        })
        emodel = EncoderModel(encoder)

        inp = tf.random.uniform((3, 9, 16), dtype=tf.float32)
        out = emodel(inp, False)
        self.assertFalse(np.isnan(out.numpy()).any())

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        emodel.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(inp, False)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def _test_masked_encoder(self, biased):
        encoder = berts.MaskedEncoder(3, {
            'model_dim': 16,
            'num_heads': 8,
            'dff': 32,
            'dropout_rate': 0.4,
            'use_bias': biased,
        })
        emodel = MaskedEncoderModel(encoder)

        inp = tf.random.uniform((3, 9, 16), dtype=tf.float32)
        mask = tf.random.uniform((3, 8, 9, 9), dtype=tf.float32)
        out = emodel(inp, mask, False)
        self.assertFalse(np.isnan(out.numpy()).any())

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        emodel.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(inp, mask, False)

        self.assertTrue(np.all(np.array_equal(out, out2)))
        shutil.rmtree(modelpath)

    def test_input_embed(self):
        """
        Test input embed
        """
        iem = berts.InputEmbed(16, {
            'maximum_position_encoding': 1000,
            'vocab_size': 200,
            'dropout_rate': 0.4,
        })
        emodel = InputEmbedModel(iem)

        inp = tf.random.uniform((3, 512), dtype=tf.int32, maxval=200, minval=0)
        out = emodel(inp, False)
        self.assertFalse(np.isnan(out.numpy()).any())

        modelpath = os.path.join('/tmp', str(uuid.uuid4()))
        emodel.save(modelpath)

        loaded = tf.keras.models.load_model(modelpath)
        out2 = loaded(inp, False)

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

    def test_perceiver_unbiased(self):
        """
        Test perceiver with use_bias set to False
        """
        self._test_perceiver(False)

    def test_perceiver_biased(self):
        """
        Test perceiver with use_bias set to True
        """
        self._test_perceiver(True)

    def test_unmasked_encoder_unbiased(self):
        """
        Test encoder with use_bias set to False
        """
        self._test_unmasked_encoder(False)

    def test_unmasked_encoder_biased(self):
        """
        Test encoder with use_bias set to True
        """
        self._test_unmasked_encoder(True)

    def test_masked_encoder_unbiased(self):
        """
        Test masked encoder with use_bias set to False
        """
        self._test_masked_encoder(False)

    def test_masked_encoder_biased(self):
        """
        Test masked encoder with use_bias set to True
        """
        self._test_masked_encoder(True)

if __name__ == '__main__':
    unittest.main()
