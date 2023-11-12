#!/usr/bin/env python3
"""
This runs pretrain_nsp.ipynb in an AWS instance
"""
# standard packages
import sys
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
S3_DIR = 'v5'
CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
ID = 'pretraining'
OUTDIR = 'out'
MODEL_ID = 'berte_pretrain'

syncer = init.S3BucketSyncer(S3_BUCKET)
syncer.download_if_notfound('configs', os.path.join(S3_DIR, ID, 's3_configs.tar.gz'))
syncer.download_if_notfound('export', os.path.join(S3_DIR, ID, 's3_export.tar.gz'))
syncer.download_if_notfound('intake', os.path.join(S3_DIR, ID, 's3_intake.tar.gz'))

# business logic
from common.pretrain import PretrainRunner, TrainingMethod

if __name__ == '__main__':
    logger = init.create_logger(ID, CLOUDWATCH_GROUP, EC2_REGION)

    logger.setLevel(logging.INFO)
    try:
        os.makedirs(OUTDIR)
    except FileExistsError:
        pass

    err = PretrainRunner('aws', CLOUDWATCH_GROUP, '',
            { 'group': CLOUDWATCH_GROUP, 'model_id': MODEL_ID },
            training_methods=[
                TrainingMethod('mlm', {
                    'metric_name': 'berte',
                    'dataset_config': "configs/mlm_dataset.yaml",
                }),
                TrainingMethod('nsp', {
                    'metric_name': 'berte_nsp',
                    'dataset_config': "configs/nsp_dataset.yaml",
                }),
            ]).\
            sequence(3, 'intake/berte_pretrain', OUTDIR)
    if err:
        print(err)
        sys.exit(1)

    syncer.tar_then_upload(OUTDIR, os.path.join(S3_DIR, ID), 'out.tar.gz')
    boto3.client('ec2', region_name=EC2_REGION).stop_instances(InstanceIds=[INSTANCE_ID])
