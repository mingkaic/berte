#!/usr/bin/env python3
"""
This runs pretrain_mlm or pretrain_nsp in an AWS instance
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
from aws_common.telemetry import get_cloudwatch_metric_reporter

_instance_info = get_instance()
S3_BUCKET = 'bidi-enc-rep-trnsformers-everywhere'
S3_DIR = 'v5.1'
CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
ID = 'pretraining'
METRIC_NAME = 'berte'
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

if __name__ == '__main__':
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    logger = init.create_logger(ID, CLOUDWATCH_GROUP, EC2_REGION)
    logger.setLevel(logging.INFO)

    PretrainerPipeline(logger, OUTDIR,
             tokenizer_model=os.path.join(IN_MODEL_DIR, "tokenizer.model")).\
    e2e(in_model_dir=IN_MODEL_DIR,
        ckpt_id=MODEL_ID,
        model_id=MODEL_ID,
        ckpt_options=tf.train.CheckpointOptions(experimental_io_device='/job:localhost'),
        report_metric=get_cloudwatch_metric_reporter(METRIC_NAME, 60))
    syncer.tar_then_upload(OUTDIR, os.path.join(S3_DIR, ID), 'out.tar.gz')
    boto3.client('ec2', region_name=EC2_REGION).stop_instances(InstanceIds=[INSTANCE_ID])
