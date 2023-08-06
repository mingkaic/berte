#!/usr/bin/env python3
"""
This module sets up training dataset
"""

import yaml

import tensorflow as tf

def setup_dataset(dataset_dirpath, training_args, tokenizer, logger=None):
    """ convert all the shards in directory path to datasets """

    def batch_process(sentences, lengths):
        return tokenizer.tokenize(sentences).to_tensor(), lengths

    try:
        dataset = (tf.data.experimental.load(dataset_dirpath)
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

    return dataset

def cache_values(cache_file, generators, *args, **kwargs):
    """
    read from cache_file if it exist and populate missing arguments from generators with args
    """
    cached_args = dict()
    try:
        with open(cache_file, "r") as file:
            args = yaml.safe_load(file.read())
            if args is not None:
                cached_args = args
    except FileNotFoundError:
        pass

    for key in generators:
        if key not in cached_args:
            cached_args[key] = generators[key](*args, **kwargs)

    with open(cache_file, "w") as file:
        file.write(yaml.dump(cached_args))

    return cached_args

def generate_dataset_width(dataset_path, tokenizer):
    """ looks up the maximum tokens length possible in the dataset """
    all_batches = tf.data.experimental.load(dataset_path)
    dataset_width = 0
    for sentence in all_batches:
        dataset_width = max(dataset_width, tokenizer.tokenize(sentence).shape[0])
    return dataset_width
