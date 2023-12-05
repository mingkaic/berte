#!/usr/bin/env python3
"""
This module includes functions for pretraining and testing nsp
"""

import numpy as np
import tensorflow as tf

import export.model as model

import common.nextsentence as ns
import common.training as training

def _dist_class(i1, i2, window_size):
    distance = abs(np.array(i1) - np.array(i2))
    distance_class = (distance / (window_size / model.DISTANCE_CLASSES)).astype(np.int32)
    return distance_class

def _to_tensor(values, pad_id):
    maxcol = np.max([np.array(row).size for row in values])
    return tf.constant(np.stack([
        np.concatenate((np.array(row), [pad_id] * (maxcol - np.array(row).size)))
        for row in values
    ]).astype(np.int32))

def mask_lm(pretrainer, tokens, mask_rate,
        donothing_prob=0.1, rand_prob=0.1):
    """ given padded tokens, return mask_rate portion of tokens """
    assert(isinstance(tokens, np.ndarray))

    metadata = pretrainer.metadata
    lengths = np.array([len(sent) for sent in tokens])
    sizes = np.floor(mask_rate * lengths).astype(np.int)

    out = []
    for (sent, size, length) in zip(tokens, sizes, lengths):
        if size < 1:
            out.append(sent)
            continue

        indices = np.random.choice(length, size=size, replace=False)
        state_probs = np.random.rand(size)

        values_info = [(
            index,
            (np.random.rand() if state_prob < donothing_prob + rand_prob else metadata.mask())
        ) for index, state_prob in zip(indices, state_probs) if state_prob < donothing_prob]
        if len(values_info) < 1:
            out.append(sent)
            continue

        indices = [index for index, _ in values_info]
        fill_values = [fill_value for _, fill_value in values_info]
        sent[indices] = fill_values
        out.append(sent)

    return out

def select_sentence_pairs(pretrainer, token_windows, batch_size, window_size):
    """ given token windows and length windows, return first and second sentences selected within the token window """

    metadata = pretrainer.metadata

    firsts = np.random.randint(low=0, high=window_size-1, size=(batch_size))
    seconds = np.array([ns.choose_second_sentence(window_size, first) for first in firsts])

    out_tokens = []
    s1_toks = []
    s2_toks = []
    length = 0
    for first_index, second_index, tokens in zip(firsts, seconds, token_windows):
        sentence1 = tokens[first_index]
        sentence2 = tokens[second_index]
        toks = np.concatenate(([metadata.cls()[0]], sentence1, [metadata.sep()], sentence2), axis=0)
        out_tokens.append(toks)
        s1_toks.append(sentence1)
        s2_toks.append(sentence2)

    return out_tokens, firsts, seconds

def build_tester(pretrainer, paragraph, logger, batch_size=15, mask_rate=0.15):
    """ build_tester returns a callable that tests the samples """
    window_size = len(paragraph)
    intokens = np.array([pretrainer.tokenize(sentence).numpy() for sentence in paragraph])
    masked = mask_lm(pretrainer, intokens, mask_rate)
    masked_window = [masked] * batch_size
    tokens, s1_indices, s2_indices =\
        select_sentence_pairs(pretrainer, masked_window, batch_size, len(paragraph))
    tokens = _to_tensor(tokens, pretrainer.metadata.pad())
    distance_class = _dist_class(s1_indices, s2_indices, window_size)

    def tester():
        prediction, _ = pretrainer.predict(tokens, training=False)
        for s1, s2, tok, pred, dist_class in zip(s1_indices, s2_indices, tokens.numpy(), prediction.numpy(), distance_class):
            labels = np.argmax(pred, axis=-1)

            logger.info('{:<25}: {}'.format("Input1", paragraph[s1]))
            logger.info('{:<25}: {}'.format("Input2", paragraph[s2]))
            logger.info('{:<25}: {}'.format("Tokens", tok))
            logger.info('{:<25}: {}'.format("Prediction Prob", pred))
            logger.info('{:<25}: {}'.format("Prediction Labels", labels))
            logger.info('{:<25}: {}'.format("Actual Dist Label", dist_class))
    return tester

NAN_LOSS_ERR_CODE = 1

def build_pretrainer(pretrainer, optimizer, training_loss, training_accuracy):
    """ build_trainer returns a callable that trains a bunch of minibatch """
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ])
    def pretrain(source, target):
        with tf.GradientTape() as tape:
            prediction, debug_info = pretrainer.predict(source, training=True)
            target = tf.cast(target, tf.int64)
            loss, debug_info2 = training.loss_function(target, prediction,
                    pad=pretrainer.metadata.pad())
            debug_info['loss'] = loss
            debug_info.update(debug_info2)

            accuracy = training.accuracy_function(target, prediction)
            gradients = tape.gradient(loss, pretrainer.trainable_variables)

        return gradients, loss, accuracy, debug_info

    def _mlm_call(batch, mask_rate):
        tokenized_batch = np.array([
            pretrainer.tokenize(sentence).numpy() for sentence in batch
        ])
        masked = mask_lm(pretrainer, tokenized_batch, mask_rate)

        pretrain_inp = _to_tensor(masked, pretrainer.metadata.pad())
        pretrain_out = _to_tensor(tokenized_batch, pretrainer.metadata.pad())

        gradients, loss, accuracy, debug_info = pretrain(pretrain_inp, pretrain_out)

        if tf.math.is_nan(loss):
            debug_info['bad_batch'] = batch
            return debug_info, NAN_LOSS_ERR_CODE

        optimizer.apply_gradients(zip(gradients, pretrainer.trainable_variables))
        training_loss(loss)
        training_accuracy(accuracy)

        return debug_info, 0

    def _nsp_call(nwindows, mask_rate):
        (batch_size, window_size) = nwindows.shape
        nwindows = [
            [pretrainer.tokenize(sentence).numpy() for sentence in window]
            for window in nwindows]
        masked_window = np.array([
            mask_lm(pretrainer, np.array(window), mask_rate)
            for window in nwindows])
        masked, s1_indices, s2_indices =\
            select_sentence_pairs(pretrainer, masked_window, batch_size, window_size)
        distance_class = _dist_class(s1_indices, s2_indices, window_size)
        batch = [
            np.concatenate(([pretrainer.metadata.cls()[dist_class]], window[s1], [pretrainer.metadata.sep()], window[s2]), axis=0)
            for dist_class, window, s1, s2 in zip(distance_class, nwindows, s1_indices, s2_indices)
        ]

        pretrain_in = _to_tensor(masked, pretrainer.metadata.pad())
        pretrain_out = _to_tensor(batch, pretrainer.metadata.pad())

        gradients, loss, accuracy, debug_info = pretrain(pretrain_in, pretrain_out)

        if tf.math.is_nan(loss):
            debug_info['bad_batch'] = batch
            return debug_info, NAN_LOSS_ERR_CODE

        optimizer.apply_gradients(zip(gradients, pretrainer.trainable_variables))
        training_loss(loss)
        training_accuracy(accuracy)

        return debug_info, 0

    return _nsp_call, _mlm_call
