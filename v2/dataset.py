#!/usr/bin/env python3
"""
This module sets up training dataset
"""

import tensorflow as tf

def setup_dataset(dataset_dirpath, training_args, tokenizer, logger=None):
    """ convert all the shards in directory path to datasets """

    try:
        dataset = tf.data.experimental.load(dataset_dirpath)
        def gen():
            for sentences in dataset:
                toks = tokenizer.tokenize(sentences)
                yield sentences, toks.shape[0]

        def batch_process(sentences, length):
            return tokenizer.tokenize(sentences).to_tensor(), length

        return (tf.data.Dataset.from_generator(gen,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ))
            .cache()
            .shuffle(training_args["buffer_size"])
            .batch(training_args["batch_size"])
            .map(batch_process)
            .prefetch(tf.data.AUTOTUNE))
    except FileNotFoundError as err:
        if logger is not None:
            logger.error('{} not found: {}'.format(dataset_dirpath, err))
    except ValueError as err:
        if logger is not None:
            logger.error('{} failed: {}'.format(dataset_dirpath, err))

    return None

def generate_dataset_width(dataset_path, tokenizer):
    """ looks up the maximum tokens length possible in the dataset """
    all_batches = tf.data.experimental.load(dataset_path)
    dataset_width = 0
    for sentence in all_batches:
        dataset_width = max(dataset_width, tokenizer.tokenize(sentence).shape[0])
    return dataset_width
