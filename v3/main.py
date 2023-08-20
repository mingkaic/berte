#!/usr/bin/env python3
"""
This runs pretrain_nsp.ipynb in an AWS instance
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
from aws_common.telemetry import get_cloudwatch_metric_reporter

_instance_info = get_instance()
INSTANCE_ID = _instance_info['instance_id']
EC2_REGION = _instance_info['ec2_region']

S3_BUCKET = 'bidi-enc-rep-trnsformers-everywhere'
S3_DIR = 'v3'
CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
ID = 'pretraining'
OUTDIR = 'out'

syncer = init.S3BucketSyncer(S3_BUCKET)
syncer.download_if_notfound('configs',
        os.path.join(S3_DIR, ID, 's3_configs.tar.gz'))
syncer.download_if_notfound('export',
        os.path.join(S3_DIR, ID, 's3_export.tar.gz'))
syncer.download_if_notfound('intake',
        os.path.join(S3_DIR, ID, 's3_intake.tar.gz'))

# business logic
from pretrain_nsp import PretrainerPipeline

if __name__ == '__main__':
    logger = init.create_logger(ID, CLOUDWATCH_GROUP, EC2_REGION)

    logger.setLevel(logging.INFO)
    try:
        os.makedirs(OUTDIR)
    except FileExistsError:
        pass

    PretrainerPipeline(logger, OUTDIR).e2e(
        nepochs=10,
        in_model_dir='intake/berte_pretrain_mlm',
        ckpt_id='berte_pretrain',
        model_id='berte_pretrain',
        ckpt_options=tf.train.CheckpointOptions(experimental_io_device='/job:localhost'),
        report_metric=get_cloudwatch_metric_reporter('berte', 60))
    syncer.tar_then_upload(OUTDIR, os.path.join(S3_DIR, ID), 'out.tar.gz')
    boto3.client('ec2', region_name=EC2_REGION).stop_instances(InstanceIds=[INSTANCE_ID])
