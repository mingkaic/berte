#!/usr/bin/env python3
"""
This tests 4_persentence_pretrain_mlm_15p.ipynb locally
"""
# standard packages
import sys
import os
import logging
import shutil

# business logic
from ps_pretrain_mlm_15p import PretrainerPipeline
from common.keras_checks import assert_model_eq

def shorten_shards(training_shards):
    """ take 1 from each shard and take the first 3 shads """
    result = dict()
    for training_key in list(training_shards.keys())[:2]:
        result[training_key] = training_shards[training_key].take(3)
    return result

if __name__ == '__main__':
    ID = 'test'
    OUTDIR = 'tmp'
    OPTIMIZER_IT = 9

    logging.basicConfig(filename="tmp/4_persentence_pretrain_mlm_15p.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    pipeline = PretrainerPipeline(logger, OUTDIR)
    # first "epoch" to generate model
    saved_pretrainer = pipeline.e2e(
            ckpt_id=ID,
            model_id=ID,
            training_preprocessing=shorten_shards,
            context_rate_overwrite=1.0) # guarantee context_rate
    model_dir = os.path.join(OUTDIR, ID)

    # check saved pretrainer
    next_pretrainer, _ = pipeline.setup_pretrainer(
        pipeline.setup_optimizer(OPTIMIZER_IT), OUTDIR, in_model_dir=model_dir)
    assert_model_eq(saved_pretrainer.preprocessor, next_pretrainer.preprocessor)
    assert_model_eq(saved_pretrainer.perceiver, next_pretrainer.perceiver)
    assert_model_eq(saved_pretrainer.latenter, next_pretrainer.latenter)
    assert_model_eq(saved_pretrainer.mask_predictor, next_pretrainer.mask_predictor)

    # second "epoch" to read model
    pipeline.e2e(
            in_model_dir=model_dir,
            ckpt_id=ID,
            model_id=ID,
            training_preprocessing=shorten_shards,
            context_rate_overwrite=1.0,
            optimizer_it=OPTIMIZER_IT) # guarantee context_rate

    shutil.rmtree(model_dir) # remove model
    shutil.rmtree(os.path.join(OUTDIR, 'checkpoints', ID)) # remove checkpoints
