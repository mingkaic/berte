#!/usr/bin/env python3
"""
This tests pretrain_nsp.ipynb locally
"""
# standard packages
import os
import logging

import shutil
import psutil

# business logic
from pretrain_nsp import PretrainerPipeline

def shorten_ds(training_ds):
    """ take 36 from the dataset """
    return training_ds.take(36)

if __name__ == '__main__':
    ID = 'test'
    OUTDIR = 'tmp'
    MODEL_DIR = 'intake/berte_pretrain_mlm'

    process = psutil.Process()
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')
    logging.basicConfig(filename="tmp/pretrain.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # second "epoch" to read model
    pipeline = PretrainerPipeline(logger, OUTDIR)
    pipeline.e2e(
            in_model_dir=MODEL_DIR,
            ckpt_id=ID,
            model_id=ID,
            training_preprocessing=shorten_ds)
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    shutil.rmtree(os.path.join(OUTDIR, 'checkpoints', ID)) # remove checkpoints
