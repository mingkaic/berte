#!/usr/bin/env python3
"""
This tests pretrain_mlm.ipynb in an AWS instance
"""
# standard packages
import logging
import os.path

# installed packages
import tensorflow as tf

# aws packages
import boto3

# local packages
import aws_common.init as init
from aws_common.instance import get_instance

_instance_info = get_instance()
INSTANCE_ID = _instance_info['instance_id']
EC2_REGION = _instance_info['ec2_region']

S3_BUCKET = 'bidi-enc-rep-trnsformers-everywhere'
S3_DIR = 'v2'
CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
ID = 'test'
OUTDIR = 'test_out'

syncer = init.S3BucketSyncer(S3_BUCKET)
syncer.download_if_notfound('configs',
        os.path.join(S3_DIR, 'pretraining', 's3_configs.tar.gz'))
syncer.download_if_notfound('export',
        os.path.join(S3_DIR, 'pretraining', 's3_export.tar.gz'))
syncer.download_if_notfound('intake',
        os.path.join(S3_DIR, 'pretraining', 's3_intake.tar.gz'))

# business logic
from pretrain_mlm import PretrainerPipeline

def shorten_ds(training_ds):
    """ take 36 from the dataset """
    return training_ds.take(36)

if __name__ == '__main__':
    logger = init.create_logger(ID, CLOUDWATCH_GROUP, EC2_REGION)
    logger.setLevel(logging.INFO)
    try:
        os.makedirs(OUTDIR)
    except FileExistsError:
        pass

    MODEL_DIR = 'intake/simpl_berte_pretrain_mlm'
    ckpt_options = tf.train.CheckpointOptions(experimental_io_device='/job:localhost')
    pipeline = PretrainerPipeline(logger, OUTDIR)
    # second "epoch" to read model
    pipeline.e2e(
            in_model_dir=MODEL_DIR,
            ckpt_id=ID,
            model_id=ID,
            training_preprocessing=shorten_ds,
            ckpt_options=ckpt_options)

    syncer.tar_then_upload(OUTDIR, os.path.join(S3_DIR, ID), 'out.tar.gz')
    boto3.client('ec2', region_name=EC2_REGION).stop_instances(InstanceIds=[INSTANCE_ID])
