#!/usr/bin/env python3
"""
This tests pretrain_mlm.ipynb locally
"""
# standard packages
import os
import logging

import shutil
import psutil

# business logic
from pretrain_mlm import PretrainerPipeline

def shorten_shards(training_shards):
    """ take 6 from each shard and take the first 3 shads """
    result = dict()
    for training_key in list(training_shards.keys())[:2]:
        result[training_key] = training_shards[training_key].take(6)
    return result

if __name__ == '__main__':
    ID = 'test'
    OUTDIR = 'tmp'
    model_dir = os.path.join(OUTDIR, ID)

    process = psutil.Process()
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')
    logging.basicConfig(filename="tmp/pretrain_mlm.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # second "epoch" to read model
    pipeline = PretrainerPipeline(logger, OUTDIR)
    pipeline.e2e(
            in_model_dir=model_dir,
            ckpt_id=ID,
            model_id=ID,
            training_preprocessing=shorten_shards)
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    shutil.rmtree(model_dir) # remove model
    shutil.rmtree(os.path.join(OUTDIR, 'checkpoints', ID)) # remove checkpoints
