#!/usr/bin/env python3
"""
This tests pretrain_mlm.ipynb locally
"""
# standard packages
import os
import logging

import psutil

# business logic
from pretrain_mlm import PretrainerPipeline
from common.keras_checks import assert_model_eq

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

    # first "epoch" to generate model
    pipeline = PretrainerPipeline(logger, OUTDIR)
    saved_pretrainer = pipeline.e2e(
            ckpt_id=ID,
            model_id=ID,
            training_preprocessing=shorten_shards)

    # check saved pretrainer
    next_pretrainer, _ = pipeline.setup_pretrainer(
        pipeline.setup_optimizer(), OUTDIR, in_model_dir=model_dir)
    assert_model_eq(saved_pretrainer.preprocessor, next_pretrainer.preprocessor)
    assert_model_eq(saved_pretrainer.mask_predictor, next_pretrainer.mask_predictor)
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')
