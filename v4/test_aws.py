#!/usr/bin/env python3
"""
This tests pretrain locally
"""
# standard packages
import os.path

import shutil
import psutil

# installed packages
# aws packages
import boto3

# local packages
import aws_common.init as init
from aws_common.instance import get_instance

S3_BUCKET = 'bidi-enc-rep-trnsformers-everywhere'
S3_DIR = 'v4'
CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
ID = 'pretraining'
OUTDIR = 'out'

syncer = init.S3BucketSyncer(S3_BUCKET)
syncer.download_if_notfound('configs', os.path.join(S3_DIR, ID, 's3_configs.tar.gz'))
syncer.download_if_notfound('export', os.path.join(S3_DIR, ID, 's3_export.tar.gz'))
syncer.download_if_notfound('intake', os.path.join(S3_DIR, ID, 's3_intake.tar.gz'))

from pretrain import PretrainRunner

if __name__ == '__main__':
    ID = 'test'
    MODEL_DIR = 'intake/berte_pretrain'

    process = psutil.Process()
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    PretrainRunner('test_aws', CLOUDWATCH_GROUP, 'berte_pretrain').sequence(5, MODEL_DIR, ID)
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    shutil.rmtree(ID) # remove checkpoints
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')
