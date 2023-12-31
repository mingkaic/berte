#!/usr/bin/env python3
"""
This runs unet_pretrain on the local machine
"""
# standard packages
import logging
import os
import shutil

# business logic
from pretrain import PretrainerPipeline

OUTDIR = 'test'
IN_MODEL_DIR = 'intake/berte_pretrain'
MODEL_ID = 'berte_pretrain'

def shorten_ds(training_ds):
    """ take 36 from the dataset """
    return training_ds.take(36)

if __name__ == '__main__':
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR, exist_ok=True)

    logging.basicConfig(filename=os.path.join(OUTDIR, "pretrain.log"),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    PretrainerPipeline(logger, OUTDIR)\
            .e2e(in_model_dir=IN_MODEL_DIR,
                    ckpt_id=MODEL_ID,
                    model_id=MODEL_ID,
                    training_preprocessing=shorten_ds)

    shutil.rmtree(OUTDIR) # remove checkpoints
