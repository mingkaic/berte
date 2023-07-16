#!/usr/bin/env python3
"""
This runs 4_persentence_pretrain_mlm_15p.ipynb in an AWS instance
"""
# standard packages
import logging
import os.path

# installed packages
import tensorflow as tf

# aws packages
import boto3

# business logic
from ps_pretrain_mlm_15p import PretrainerPipeline

# local packages
import aws_common.init as init
from aws_common.instance import get_instance
from aws_common.telemetry import get_cloudwatch_metric_reporter

if __name__ == '__main__':
    _instance_info = get_instance()
    INSTANCE_ID = _instance_info['instance_id']
    EC2_REGION = _instance_info['ec2_region']

    S3_BUCKET = 'bidi-enc-rep-trnsformers-everywhere'
    S3_DIR = 'v1/pretraining'
    CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
    ID = '4_persentence_pretrain_mlm_15p'
    OUTDIR = 'out'

    syncer = init.S3BucketSyncer(S3_BUCKET)
    syncer.download_if_notfound('configs',
            os.path.join(S3_DIR, ID, 's3_configs.tar.gz'))
    syncer.download_if_notfound('export',
            os.path.join(S3_DIR, ID, 's3_export.tar.gz'))

    logger = init.create_logger(ID, CLOUDWATCH_GROUP, EC2_REGION)

    logger.setLevel(logging.INFO)
    try:
        os.makedirs(OUTDIR)
    except FileExistsError:
        pass

    PretrainerPipeline(logger, OUTDIR).e2e(
        nepochs=5,
        in_model_dir='export/berte_pretrain_mlm_15p',
        ckpt_id='train_15p_ps_cont',
        model_id='berte_pretrain_mlm_15p_cont',
        ckpt_options=tf.train.CheckpointOptions(experimental_io_device='/job:localhost'),
        report_metric=get_cloudwatch_metric_reporter('berte', 60),
        optimizer_it=269150)
    syncer.tar_then_upload(OUTDIR, os.path.join(S3_DIR, ID), 'out.tar.gz')
    boto3.client('ec2', region_name=EC2_REGION).stop_instances(InstanceIds=[INSTANCE_ID])
