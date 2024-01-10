#!/usr/bin/env python3
"""
This runs pretrain_unetprepare on the aws machine on test mode.
No metric reporting.
"""
# standard packages
import logging
import os

# installed packages
import tensorflow as tf

# aws packages
import boto3

# local packages
import aws_common.init as init
from aws_common.instance import get_instance

_instance_info = get_instance()
S3_BUCKET = 'bidi-enc-rep-trnsformers-everywhere'
S3_DIR = 'v5.2'
CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
ID = 'pretraining'
INSTANCE_ID = _instance_info['instance_id']
EC2_REGION = _instance_info['ec2_region']

syncer = init.S3BucketSyncer(S3_BUCKET)
syncer.download_if_notfound('configs', os.path.join(S3_DIR, ID, 's3_configs.tar.gz'))
syncer.download_if_notfound('export', os.path.join(S3_DIR, ID, 's3_export.tar.gz'))
syncer.download_if_notfound('intake', os.path.join(S3_DIR, ID, 's3_intake.tar.gz'))

# business logic
from pretrain import PretrainerPipeline

IN_MODEL_DIR = 'intake/berte_pretrain'
OUTDIR = 'out'
MODEL_ID = 'berte_pretrain'

def shorten_ds(training_ds):
    """ take 36 from the dataset """
    return training_ds.take(36)

if __name__ == '__main__':
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    logger = init.create_logger(ID, CLOUDWATCH_GROUP, EC2_REGION)
    logger.setLevel(logging.INFO)

    PretrainerPipeline(logger, OUTDIR)\
            .e2e(in_model_dir=IN_MODEL_DIR,
                 ckpt_id=MODEL_ID,
                 model_id=MODEL_ID,
                 ckpt_options=tf.train.CheckpointOptions(experimental_io_device='/job:localhost'),
                 training_preprocessing=shorten_ds)
    boto3.client('ec2', region_name=EC2_REGION).stop_instances(InstanceIds=[INSTANCE_ID])
