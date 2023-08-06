#!/usr/bin/env python3
"""
This tests pretrain_mlm.ipynb in an AWS instance
"""
# standard packages
import logging
import os.path

# installed packages
import tensorflow as tf

# business logic
from pretrain_mlm import PretrainerPipeline
from common.keras_checks import assert_model_eq

# local packages
import aws_common.init as init
from aws_common.instance import get_instance

def shorten_shards(training_shards):
    """ take 10 from each shard (approx 26 shards) """
    result = dict()
    for training_key in training_shards.keys():
        result[training_key] = training_shards[training_key].take(10)
    return result

if __name__ == '__main__':
    _instance_info = get_instance()
    INSTANCE_ID = _instance_info['instance_id']
    EC2_REGION = _instance_info['ec2_region']

    S3_BUCKET = 'bidi-enc-rep-trnsformers-everywhere'
    S3_DIR = 'v1/pretraining'
    CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
    ID = 'test'
    OUTDIR = 'test_out'

    syncer = init.S3BucketSyncer(S3_BUCKET)
    syncer.download_if_notfound('configs',
            os.path.join(S3_DIR, '5_pretrain_mlm', 's3_configs.tar.gz'))
    syncer.download_if_notfound('export',
            os.path.join(S3_DIR, '5_pretrain_mlm', 's3_export.tar.gz'))

    logger = init.create_logger(ID, CLOUDWATCH_GROUP, EC2_REGION)
    logger.setLevel(logging.INFO)
    try:
        os.makedirs(OUTDIR)
    except FileExistsError:
        pass

    ckpt_options = tf.train.CheckpointOptions(experimental_io_device='/job:localhost')
    pipeline = PretrainerPipeline(logger, OUTDIR)
    # first "epoch" to generate model
    saved_pretrainer = pipeline.e2e(
            ckpt_id=ID,
            model_id=ID,
            training_preprocessing=shorten_shards,
            ckpt_options=ckpt_options)
    model_dir = os.path.join(OUTDIR, ID)

    # check saved pretrainer
    next_pretrainer, _ = pipeline.setup_pretrainer(pipeline.setup_optimizer(), OUTDIR,
            in_model_dir=model_dir)
    assert_model_eq(saved_pretrainer.preprocessor, next_pretrainer.preprocessor)
    assert_model_eq(saved_pretrainer.mask_predictor, next_pretrainer.mask_predictor)
