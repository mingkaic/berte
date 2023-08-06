#!/usr/bin/env python3
"""
This module includes functions for pretraining and testing mlm
"""

import numpy as np
import tensorflow as tf

import common.training as training

def mask_lm(tokens, lengths, mask_rate, pad_id, mask_id, ds_width,
        donothing_prob=0.1, rand_prob=0.1):
    """ given padded tokens, return mask_rate portion of tokens """
    if isinstance(lengths, tf.Tensor):
        lengths = lengths.numpy()
    lengths -= 2 # exclude bos and eos

    # ensure tokens has shape [batch_size, ds_width]
    tokens = tf.pad(tokens, ([0, 0], [0, ds_width - tokens.numpy().shape[1]]),
        constant_values=pad_id)

    state_prob = np.random.rand()

    if state_prob < donothing_prob:
        return tokens, tokens

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
    return tokens, masked_tokens

def build_tester(action_module, samples, mask_sizes, logger):
    """ build_tester returns a callable that tests the samples """
    n_extras = mask_sizes[0]-len(samples)
    sentences = tf.constant(samples + [""]*n_extras)
    tokens = action_module.tokenize(sentences)
    lengths = np.array([len(token) for token in tokens.numpy()])
    _, masked = mask_lm(tokens, lengths, 0.1,
        pad_id=action_module.metadata.pad(),
        mask_id=action_module.metadata.mask(),
        ds_width=mask_sizes[1])

    def tester():
        prediction, _ = action_module.mask_prediction(masked, training=False)
        for i, sentence in enumerate(samples):
            local_toks = tokens[i, :]
            local_preds = prediction[i, :, :].numpy()
            length = lengths[i]

            total_probs = tf.reduce_sum(local_preds, axis=-1)[:length]

            true_preds = [local_preds[j, local_tok] for j, local_tok in enumerate(local_toks)]
            guess_preds = [local_preds[j, guess_tok]
                for j, guess_tok in enumerate(tf.argmax(prediction[i, :, :], axis=-1))][:length]

            logger.info('{:<15}: {}'.format("Input", sentence))
            logger.info('{:<15}: {}'.format("Tokens", local_toks.numpy()))
            logger.info('{:<15}: {}'.format("Truth prob %", true_preds))
            logger.info('{:<15}: {}'.format("Best prob %", guess_preds))
            logger.info('{:<15}: {}'.format("Total prob", total_probs))
    return tester

NAN_LOSS_ERR_CODE = 1

def build_pretrainer(action_module, optimizer, ds_width):
    """ build_trainer returns a callable that trains a bunch of minibatch """
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    batch_shape = (None, ds_width)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=batch_shape, dtype=tf.int32),
        tf.TensorSpec(shape=batch_shape, dtype=tf.int32),
    ])
    def pretrain(source, target):
        with tf.GradientTape() as tape:
            prediction, debug_info = action_module.mask_prediction(source, training=True)
            target = tf.cast(target, tf.int64)
            loss, debug_info2 = training.loss_function(target, prediction,
                    pad=action_module.metadata.pad())
            debug_info['loss'] = loss
            debug_info.update(debug_info2)

            accuracy = training.accuracy_function(target, prediction)
            gradients = tape.gradient(loss, action_module.trainable_variables)

        return gradients, loss, accuracy, debug_info

    def _call(batch, lengths, mask_rate):
        batch, masked = mask_lm(batch, lengths, mask_rate,
                pad_id=action_module.metadata.pad(),
                mask_id=action_module.metadata.mask(),
                ds_width=ds_width)
        gradients, loss, accuracy, debug_info = pretrain(masked, batch)

        if tf.math.is_nan(loss):
            debug_info['bad_batch'] = batch
            return debug_info, NAN_LOSS_ERR_CODE

        optimizer.apply_gradients(zip(gradients, action_module.trainable_variables))
        train_loss(loss)
        train_accuracy(accuracy)

        return debug_info, 0

    return _call, train_loss, train_accuracy
