#!/usr/bin/env python3
"""
This runs pretrain_nsp.ipynb in an AWS instance
"""
# standard packages
import logging
import os.path

# installed packages
# aws packages
import boto3

# local packages
import aws_common.init as init
from aws_common.instance import get_instance

_instance_info = get_instance()
INSTANCE_ID = _instance_info['instance_id']
EC2_REGION = _instance_info['ec2_region']

S3_BUCKET = 'bidi-enc-rep-trnsformers-everywhere'
S3_DIR = 'v4'
CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
ID = 'pretraining'
OUTDIR = 'out'

syncer = init.S3BucketSyncer(S3_BUCKET)
syncer.download_if_notfound('configs', os.path.join(S3_DIR, ID, 's3_configs.tar.gz'))
syncer.download_if_notfound('export', os.path.join(S3_DIR, ID, 's3_export.tar.gz'))
syncer.download_if_notfound('intake', os.path.join(S3_DIR, ID, 's3_intake.tar.gz'))

# business logic
from pretrain import PretrainRunner

if __name__ == '__main__':
    logger = init.create_logger(ID, CLOUDWATCH_GROUP, EC2_REGION)

    logger.setLevel(logging.INFO)
    try:
        os.makedirs(OUTDIR)
    except FileExistsError:
        pass

    PretrainRunner('aws', CLOUDWATCH_GROUP, 'berte_pretrain').\
            sequence(10, 'intake/berte_pretrain', OUTDIR)
    syncer.tar_then_upload(OUTDIR, os.path.join(S3_DIR, ID), 'out.tar.gz')
    boto3.client('ec2', region_name=EC2_REGION).stop_instances(InstanceIds=[INSTANCE_ID])
