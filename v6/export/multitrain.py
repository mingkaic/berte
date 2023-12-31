#!/usr/bin/env python3
"""
This module includes functions for pretraining and testing mlm and nsp
"""

import numpy as np
import tensorflow as tf

import common.training as training
import common.nextsentence as ns

def mask_lm(tokens, lengths, mask_rate, mask_id,
        donothing_prob=0.1, rand_prob=0.1):
    """ given padded tokens, return mask_rate portion of tokens """
    state_prob = np.random.rand()
    if state_prob < donothing_prob:
        return tokens

    if isinstance(lengths, tf.Tensor):
        lengths = lengths.numpy()
    lengths -= 2 # exclude bos and eos

    num_masked = np.maximum(1, mask_rate * lengths).astype(int)
    npmask = np.zeros(tokens.shape)
    for i, (length, size) in enumerate(zip(lengths, num_masked)):
        if length > 0:
            indices = np.random.choice(length, size=size, replace=False) + 1 # after bos
            npmask[i, indices] = 1
    mask = tf.constant(npmask, dtype=tf.int32)

    if state_prob < donothing_prob + rand_prob:
        fill_values = np.random.rand(*tokens.shape)
    else:
        fill_values = mask_id

    masked_tokens = tokens * (1 - mask) + fill_values * mask
    return masked_tokens

def select_sentence_pairs(action_module, windows):
    (batch_size, window_size) = tf.shape(windows).numpy()

    firsts = np.random.randint(low=0, high=window_size-1, size=(batch_size))
    seconds = np.array([ns.choose_second_sentence(window_size, first) for first in firsts])

    metadata = action_module.metadata
    tokens = []
    length = 0
    for first_index, second_index, window in zip(firsts, seconds, windows.numpy()):
        sentence1 = action_module.tokenizer.tokenize(window[first_index])
        sentence2 = action_module.tokenizer.tokenize(window[second_index])
        toks = tf.concat((sentence1, [metadata.sep()], sentence2), axis=0)
        length = max(length, tf.shape(toks)[0].numpy())
        tokens.append(toks)

    results = tf.stack([
        tf.concat((toks, (length - tf.shape(toks)[0].numpy()) * [metadata.pad()]), axis=0)
        for toks in tokens])
    return results, firsts, seconds

NAN_LOSS_ERR_CODE = 1

mse = tf.keras.losses.MeanSquaredError()

def build_pretrainer(action_module, optimizer):
    """ build_trainer returns a callable that trains a bunch of minibatch """
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),

        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None), dtype=tf.float32),
    ])
    def pretrain(mask_source, mask_target, ns_source, ns_target):
        with tf.GradientTape() as tape:
            mask_prediction, debug_info = action_module.mask_prediction(mask_source, training=True)
            ns_prediction, debug_info2 = action_module.ns_prediction(ns_source)

            mask_target = tf.cast(mask_target, tf.int64)
            ns_target = tf.cast(ns_target, tf.int64)

            mask_loss, debug_info3 = training.loss_function(
                    mask_target,
                    mask_prediction,
                    pad=action_module.metadata.pad())
            ns_loss = mse(ns_target, ns_prediction)

            # give them equal weight
            loss = (mask_loss + ns_loss) / 2

            debug_info.update(debug_info2)
            debug_info.update({
                'loss': loss,
                'mask_loss': mask_loss,
                'ns_loss': ns_loss,
            })
            mask_accuracy = training.accuracy_function(target, prediction)

            gradients = tape.gradient(loss, action_module.trainable_variables)

        return gradients, loss, mask_accuracy, debug_info

    def _call(maskbatch, lengths, mask_rate, windows):
        masked = mask_lm(
                maskbatch, lengths, mask_rate,
                mask_id=action_module.metadata.mask())
        nsbatch, firsts, seconds = select_sentence_pairs(
                action_module, windows)
        gradients, loss, mask_accuracy, debug_info = pretrain(
                masked, maskbatch,
                nsbatch, ns.class_reducer(distances))

        if tf.math.is_nan(loss):
            debug_info['bad_batch'] = batch
            return debug_info, NAN_LOSS_ERR_CODE

        optimizer.apply_gradients(zip(gradients, action_module.trainable_variables))
        train_loss(loss)
        train_accuracy(mask_accuracy)

        return debug_info, 0

    return _call, train_loss, train_accuracy
