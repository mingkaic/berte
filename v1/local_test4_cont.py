#!/usr/bin/env python3
"""
This tests 4_persentence_pretrain_mlm_15p.ipynb locally with existing model
"""
# standard packages
import os
import sys
import logging
import shutil

# business logic
from ps_pretrain_mlm_15p import PretrainerPipeline

def shorten_shards(training_shards):
    """ take 1 from each shard and take the first 3 shads """
    result = dict()
    for training_key in list(training_shards.keys())[:2]:
        result[training_key] = training_shards[training_key].take(1)
    return result

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    PretrainerPipeline(logger, 'tmp').e2e(
            in_model_dir='export/berte_pretrain_mlm_15p',
            ckpt_id='test',
            model_id='test',
            training_preprocessing=shorten_shards,
            context_rate_overwrite=1.0, # guarantee context_rate
            optimizer_it=269150)
    shutil.rmtree('tmp/test') # remove model
    shutil.rmtree('tmp/checkpoints/test') # remove checkpoints
