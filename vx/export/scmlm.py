#!/usr/bin/env python3
"""
This module includes functions for pretraining and testing nsp
"""

import tensorflow as tf

import export.mlm as mlm
import common.training as training

def build_pretrainer(preprocessor, action_module, optimizer, ds_width, paragraph_length):
    """ build_trainer returns a callable that trains a bunch of minibatch """
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    def pretrain(source, target, *args):
        enc, latent, debug_info = preprocessor.preprocess(source, training=False)

        with tf.GradientTape() as tape:
            prediction, debug_info2 = action_module.multi_latent_ml_prediction(
                    enc, latent, training=True, *args)
            target = tf.cast(target, tf.int64)
            loss, debug_info3 = training.loss_function(target, prediction,
                    pad=action_module.metadata.pad())

            debug_info['loss'] = loss
            debug_info.update(debug_info2)
            debug_info.update(debug_info3)

            accuracy = training.accuracy_function(target, prediction)
            gradients = tape.gradient(loss, action_module.trainable_variables)

        return gradients, loss, accuracy, debug_info

    def _call(batch, lengths, mask_rate):
        latent = None
        for i in range(paragraph_length):
            mask = lengths[:, i] > 0
            batch = tf.boolean_mask(batch, mask, axis=0)
            lengths = tf.boolean_mask(lengths, mask, axis=0)
            target, source = mlm.mask_lm(
                    preprocessor.tokenize(batch[:, i]), lengths[:, i], mask_rate,
                    pad_id=preprocessor.metadata.pad(),
                    mask_id=preprocessor.metadata.mask(),
                    ds_width=ds_width)
            if latent is not None:
                latent = tf.boolean_mask(latent, mask, axis=0)
                latent = tf.stop_gradient(latent)
                gradients, loss, accuracy, debug_info = pretrain(source, target, latent)
            else:
                gradients, loss, accuracy, debug_info = pretrain(source, target)
            latent = debug_info['ml_prediction.perceiver']
            if tf.math.is_nan(loss):
                debug_info['bad_batch'] = batch
                return debug_info, mlm.NAN_LOSS_ERR_CODE
            optimizer.apply_gradients(zip(gradients, action_module.trainable_variables))
            train_loss(loss)
            train_accuracy(accuracy)

        return debug_info, 0

    return _call, train_loss, train_accuracy
