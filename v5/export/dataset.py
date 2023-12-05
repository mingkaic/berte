#!/usr/bin/env python3
"""
This module sets up training dataset
"""

import tensorflow as tf

def setup_mlm_dataset(dataset_dirpath, training_args, logger=None):
    """ convert all the shards in directory path to datasets """

    try:
        return (tf.data.experimental.load(dataset_dirpath)
            .cache()
            .shuffle(training_args["buffer_size"])
            .batch(training_args["batch_size"])
            .prefetch(tf.data.AUTOTUNE))
    except FileNotFoundError as err:
        if logger is not None:
            logger.error('{} not found: {}'.format(dataset_dirpath, err))
    except ValueError as err:
        if logger is not None:
            logger.error('{} failed: {}'.format(dataset_dirpath, err))

    return None

def setup_nsp_dataset(dataset_dirpath, training_args, logger=None):
    """ convert all the shards in directory path to datasets """

    window_size = training_args["window"]
    try:
        dataset = tf.data.experimental.load(dataset_dirpath)
        def gen():
            for sentences in dataset:
                toks = tokenizer.tokenize(sentences)
                yield sentences, toks.shape[0]

        return (tf.data.experimental.load(dataset_dirpath)
            .cache()
            .unbatch()
            .window(window_size, shift=1, drop_remainder=True)
            .flat_map(lambda x: x.batch(window_size))
            .shuffle(training_args["buffer_size"])
            .batch(training_args["batch_size"])
            .prefetch(tf.data.AUTOTUNE))
    except FileNotFoundError as err:
        if logger is not None:
            logger.error('{} not found: {}'.format(dataset_dirpath, err))
    except ValueError as err:
        if logger is not None:
            logger.error('{} failed: {}'.format(dataset_dirpath, err))

    return None
