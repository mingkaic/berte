#!/usr/bin/env python3
"""
This module sets up training dataset
"""

import os
import yaml

import tensorflow as tf

def setup_shards(shard_dirpath, training_args, tokenizer, logger=None):
    """ convert all the shards in directory path to datasets """

    def batch_process(sentences, lengths):
        return tokenizer.tokenize(sentences).to_tensor(), lengths

    training_shards = dict()
    training_keys = os.listdir(shard_dirpath)
    training_keys.sort()
    for shard in training_keys:
        try:
            training_shards[shard] = (tf.data.experimental.load(shard_dirpath + shard)
                                                          .cache()
                                                          .shuffle(training_args["buffer_size"])
                                                          .batch(training_args["batch_size"])
                                                          .map(batch_process)
                                                          .prefetch(tf.data.AUTOTUNE))
        except FileNotFoundError as err:
            if logger is not None:
                logger.error('{} not found: {}'.format(shard, err))
        except ValueError as err:
            if logger is not None:
                logger.error('{} failed: {}'.format(shard, err))

    return training_shards, training_keys

def generate_dataset_width(dataset_path, tokenizer):
    """ looks up the maximum tokens length possible in the dataset """
    all_batches = tf.data.experimental.load(dataset_path)
    dataset_width = 0
    for sentence in all_batches:
        dataset_width = max(dataset_width, tokenizer.tokenize(sentence).shape[0])
    return dataset_width
