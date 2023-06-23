#!/usr/bin/env python3
"""
This module contains all logic for aws specific deployments
"""
import tempfile
import tarfile
import os.path
import logging

from cloudwatch import cloudwatch
import boto3

class S3BucketSyncer:
    """ Sync S3 assets and local assets for a specific bucket """
    def __init__(self, bucket):
        self.client = boto3.client('s3')
        self.bucket = bucket

    def download_if_notfound(self, dirpath, key):
        """ download an asset from s3 if not exist """
        if not os.path.exists(dirpath):
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            with tempfile.TemporaryFile(mode='wb') as tempf:
                tempf.write(resp['Body'].read())
                res_tar = tarfile.open(tempf.name)
                res_tar.extractall()
                res_tar.close()

    def tar_then_upload(self, dirpath, parent_key, name):
        """ tars dirpath then write to s3 """
        with tarfile.open(name, 'w:gz') as tar:
            tar.add(dirpath, arcname=os.path.basename(dirpath))
        self.client.upload_file(name, Bucket=self.bucket, Key=os.path.join(parent_key, name))

def create_logger(log_id, group, region):
    """ return cloudwatch logger """
    logger = logging.getLogger(log_id)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s - %(message)s')
    handler = cloudwatch.CloudwatchHandler(log_group=group, region=region)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
