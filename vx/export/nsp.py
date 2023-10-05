#!/usr/bin/env python3
"""
This module includes functions for pretraining and testing nsp
"""

import numpy as np
import tensorflow as tf

import export.mlm as mlm

import common.nextsentence as ns

def select_sentence_pairs(preprocessor, windows):
    (batch_size, window_size) = tf.shape(windows).numpy()

    firsts = np.random.randint(low=0, high=window_size-1, size=(batch_size))
    seconds = np.array([ns.choose_second_sentence(window_size, first) for first in firsts])

    metadata = preprocessor.metadata
    tokens = []
    length = 0
    for first_index, second_index, window in zip(firsts, seconds, windows.numpy()):
        sentence1 = preprocessor.tokenizer.tokenize(window[first_index])
        sentence2 = preprocessor.tokenizer.tokenize(window[second_index])
        toks = tf.concat((sentence1, [metadata.sep()], sentence2), axis=0)
        length = max(length, tf.shape(toks)[0].numpy())
        tokens.append(toks)

    results = tf.stack([
        tf.concat((toks, (length - tf.shape(toks)[0].numpy()) * [metadata.pad()]), axis=0)
        for toks in tokens])
    return results, firsts, seconds

def build_tester(preprocessor, action_module, paragraph, logger):
    """ build_tester returns a callable that tests the samples """
    tokens = preprocessor.tokenize(paragraph)
    tokens, firsts, seconds = select_sentence_pairs(
            preprocessor, tf.stack([paragraph] * 15))
    enc, latent, _ = preprocessor.preprocess(tokens, training=False)

    def tester():
        labels, _ = action_module.ns_prediction(enc, latent, training=False)
        for first, second, tok, label in zip(firsts, seconds, tokens.numpy(), labels.numpy()):
            dist = abs(first - second)
            label_class = ns.class_reducer(dist)

            logger.info('{:<15}: {}'.format("Input1", paragraph[first]))
            logger.info('{:<15}: {}'.format("Input2", paragraph[second]))
            logger.info('{:<15}: {}'.format("Tokens", tok))
            logger.info('{:<15}: {}'.format("Distance", dist))
            logger.info('{:<15}: {}'.format("Guess Label", label))
            logger.info('{:<15}: {}'.format("Actual Label", label_class))
    return tester

mse = tf.keras.losses.MeanSquaredError()

def build_pretrainer(preprocessor, action_module, optimizer):
    """ build_trainer returns a callable that trains a bunch of minibatch """
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None), dtype=tf.float32),
    ])
    def pretrain(source, target):
        enc, latent, debug_info = preprocessor.preprocess(source, training=False)

        with tf.GradientTape() as tape:
            prediction, debug_info2 = action_module.ns_prediction(enc, latent, training=True)
            loss = mse(target, prediction)

            debug_info['loss'] = loss
            debug_info.update(debug_info2)

            gradients = tape.gradient(loss, action_module.trainable_variables)

        return gradients, loss, debug_info

    def _call(windows):
        batch, firsts, seconds = select_sentence_pairs(preprocessor, windows)
        distances = abs(firsts - seconds)
        gradients, loss, debug_info = pretrain(batch, ns.class_reducer(distances))

        if tf.math.is_nan(loss):
            debug_info['bad_batch'] = batch
            return debug_info, mlm.NAN_LOSS_ERR_CODE

        optimizer.apply_gradients(zip(gradients, action_module.trainable_variables))
        train_loss(loss)

        return debug_info, 0

    return _call, train_loss
