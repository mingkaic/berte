#!/usr/bin/env python3
"""
This module includes functions for training and testing
"""

import math
import functools

import numpy as np
import tensorflow as tf

import telemetry
import common.training as training

def mask_lm(tokens, lengths, mask_rate, mask_sizes, pad):
    """ given padded tokens, return mask_rate portion of tokens """
    if isinstance(lengths, tf.Tensor):
        lengths = lengths.numpy()
    lengths -= 2 # exclude bos and eos

    num_masked = np.maximum(1, mask_rate * lengths).astype(int)
    npmask = np.zeros(mask_sizes)
    for i, (length, size) in enumerate(zip(lengths, num_masked)):
        if length > 0:
            indices = np.random.choice(length, size=size, replace=False) + 1 # after bos
            npmask[i, indices] = 1

    tokens = tf.pad(tokens, ([0, 0], [0, mask_sizes[1] - tokens.numpy().shape[1]]),
            constant_values=pad)
    mask = tf.constant(npmask, dtype=tf.int32)
    not_mask = tf.abs(1 - mask)
    return tokens, tokens * not_mask, tokens * mask

def build_tester(action_module, samples, mask_sizes, logger):
    """ build_tester returns a callable that tests the samples """
    n_extras = mask_sizes[0]-len(samples)
    sentences = tf.constant(samples + [""]*n_extras)
    tokens = action_module.tokenize(sentences)
    lengths = np.array([len(token) for token in tokens.numpy()])
    _, masked_tokens, masked = mask_lm(tokens, lengths, 0.1,
            mask_sizes=mask_sizes,
            pad=action_module.metadata["PAD"])

    def tester():
        prediction = action_module.mask_prediction(masked_tokens, training=False)
        for i, sentence in enumerate(samples):
            localmask = masked[i, :]
            maskidx = tf.argmax(localmask)
            masktok = localmask[maskidx]
            localpred = prediction[i, maskidx, :]
            guesstok = tf.argmax(localpred)
            guess_pred = localpred[guesstok]
            true_pred = localpred[masktok]

            logger.info('{:<15}: {}'.format("Input", sentence))
            logger.info('{:<15}: {}'.format("Tokens", tokens[i].numpy()))
            logger.info('{:<15}: {}'.format("Mask Index", maskidx))
            logger.info('{:<15}: {}'.format("Mask Token", masktok))
            logger.info('{:<15}: {}'.format("Mask Pred %", true_pred))
            logger.info('{:<15}: {}'.format("Pred Token", guesstok))
            logger.info('{:<15}: {}'.format("Pred %", guess_pred))
    return tester

NAN_LOSS_ERR_CODE = 1

UNSTABLE_LOSS_ERR_CODE = 2

def build_pretrainer(action_module, optimizer, batch_shape):
    """ build_trainer returns a callable that trains a batch """
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function(input_signature=[
        tf.TensorSpec(shape=batch_shape, dtype=tf.int32),
        tf.TensorSpec(shape=batch_shape, dtype=tf.int32),
        tf.TensorSpec(shape=batch_shape, dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ])
    def contexted_persentence_mlm_pretrain(orig, source, target, worst_loss):
        with tf.GradientTape() as tape:
            _, lat, debug_info = action_module.latent_prediction(orig, training=True)
            prediction, debug_info2 = action_module.mask_prediction(source, latent_pred=lat, training=True)
            loss = training.loss_function(target, prediction, pad=action_module.metadata["PAD"])
            debug_info.update(debug_info2)
            debug_info['loss'] = loss
            trainable_vars = action_module.contexted_trainable_variables()

        if tf.math.is_nan(loss):
            return debug_info, NAN_LOSS_ERR_CODE # return instead of raise, since tf has issues

        if loss > worst_loss: # skip bad batches
            return debug_info, UNSTABLE_LOSS_ERR_CODE # return instead of raise, since tf has issues

        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        train_loss(loss)
        train_accuracy(training.accuracy_function(target, prediction))
        return debug_info, 0

    @tf.function(input_signature=[
        tf.TensorSpec(shape=batch_shape, dtype=tf.int32),
        tf.TensorSpec(shape=batch_shape, dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ])
    def uncontexted_persentence_mlm_pretrain(source, target, worst_loss):
        with tf.GradientTape() as tape:
            prediction, debug_info = action_module.mask_prediction(source, training=True)
            loss = training.loss_function(target, prediction, pad=action_module.metadata["PAD"])
            debug_info['loss'] = loss
            trainable_vars = action_module.uncontexted_trainable_variables()

        if tf.math.is_nan(loss):
            return debug_info, NAN_LOSS_ERR_CODE # return instead of raise, since tf has issues

        if loss > worst_loss: # skip bad batches
            return debug_info, UNSTABLE_LOSS_ERR_CODE # return instead of raise, since tf has issues

        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        train_loss(loss)
        train_accuracy(training.accuracy_function(target, prediction))
        return debug_info, 0

    def pretrainer(batch, lengths, prev_loss, mask_rate, context_rate):
        orig, source, target = mask_lm(batch, lengths, mask_rate,
                mask_sizes=batch_shape,
                pad=action_module.metadata["PAD"])
        target = tf.cast(target, tf.int64)
        if np.random.rand() < context_rate:
            return contexted_persentence_mlm_pretrain(orig, source, target, prev_loss)
        return uncontexted_persentence_mlm_pretrain(source, target, prev_loss)

    return pretrainer, train_loss, train_accuracy

SKIP_LOSS = tf.constant(math.inf, dtype=tf.float32)

EPOCH_PRETRAINER_ARGS = [
    "training_settings",
    "training_loss",
    "training_accuracy",
    "training_shards",
    "training_cb",
    "ckpt_save_cb",
    "bucket",
    "prev_losses",
]

class EpochPretrainerInitBuilder:
    """ build pretrainer init arguments """
    def __init__(self):
        self.arg = dict()
        for key in EPOCH_PRETRAINER_ARGS:
            setattr(self.__class__, key, functools.partial(self.add_arg, key=key))

    def add_arg(self, value, key):
        """ add custom arg """
        self.arg[key] = value
        return self

    def build(self):
        """ return built arg """
        return self.arg

class EpochPretrainer:
    """ Reusable pretrainer for running training epochs """
    def __init__(self, args):

        self.args = args
        self.training_shards = args["training_shards"]
        self.training_keys = list(self.training_shards.keys())
        self.training_keys.sort()

        self.bucket = args["bucket"]
        self.prev_losses = args["prev_losses"]

        self.training_loss = args["training_loss"]
        self.training_accuracy = args["training_accuracy"]

    def loss_check(self, default_value=SKIP_LOSS):
        """ determine whether to take the default_value or return using prev_loss Window """
        if self.bucket.can_skip() or not self.bucket.have_quota():
            # warming up or ran out of quota
            return default_value
        return self.prev_losses(default_value=default_value,
                stdev_coeff=self.args["training_settings"]["skip_bad_loss"]["stdev_coeff"])

    def run_epoch(self, skip_shards=None, logger=None, nan_reporter=None):
        """ run a single epoch """
        if logger is None:
            logger = telemetry.EmptyLogger()

        self.training_loss.reset_states()
        self.training_accuracy.reset_states()
        nskipped = 0
        batch = 0
        for (shard, training_key) in enumerate(self.training_keys):
            logger.info('Training shard %s', training_key)
            if skip_shards is  not None and training_key in skip_shards:
                skip_it = self.training_shards[training_key].cardinality()
                logger.info('skipped %s: fastforward %d iterations', training_key, skip_it)
                batch += skip_it
                self.bucket.iter += skip_it
                skip_shards.remove(training_key)
                continue

            for (sentence, lengths) in self.training_shards[training_key]:
                check = self.loss_check()
                debug_info, err_code = self.args["training_cb"](sentence, lengths, check)
                loss = debug_info['loss']
                skipped = False
                if err_code != 0:
                    if err_code == NAN_LOSS_ERR_CODE:
                        if nan_reporter is not None:
                            nan_reporter(sentence, debug_info)
                        logger.error('batch %d produced nan loss! skipping...', batch)
                    loss = check
                    skipped = True

                # update quota
                self.bucket.update(skipped)
                if skipped:
                    nskipped += 1
                else:
                    # update window
                    self.prev_losses.add(loss.numpy())

                if batch % 50 == 0:
                    logger.info('Batch %d Loss %.4f Accuracy %.4f Skipped %d',
                            batch,
                            self.training_loss.result(),
                            self.training_accuracy.result(),
                            nskipped)
                    nskipped = 0

                batch += 1

            if ((shard + 1) % self.args["training_settings"]["shards_per_save"] == 0):
                logger.info('Saving checkpoint for shard %d at %s',
                        shard+1, self.args["ckpt_save_cb"](None))
