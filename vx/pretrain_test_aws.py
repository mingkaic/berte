#!/usr/bin/env python3
"""
This runs pretrain_mlm or pretrain_nsp on the aws machine on test mode.
No metric reporting.
"""
# standard packages
import logging
import os.path
import argparse

# installed packages
import tensorflow as tf

# business logic
from pretrain_mlm import PretrainerPipeline as MLMPretrainerPipeline
from pretrain_nsp import PretrainerPipeline as NSPPretrainerPipeline
from pretrain_scmlm import PretrainerPipeline as SCMLMPretrainerPipeline

# local packages
import aws_common.init as init
from aws_common.instance import get_instance
from aws_common.telemetry import get_cloudwatch_metric_reporter

_instance_info = get_instance()
INSTANCE_ID = _instance_info['instance_id']
EC2_REGION = _instance_info['ec2_region']
ID = 'pretraining'

def shorten_ds(training_ds):
    """ take 36 from the dataset """
    return training_ds.take(36)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='pretrain_test_aws',
                    description='runs pretrain_mlm on the local environment')
    parser.add_argument('type')
    parser.add_argument('in_model_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--group')
    parser.add_argument('--model_id')
    parser.add_argument('--metric_name', default='berte')
    args = parser.parse_args()

    logger = init.create_logger(ID, args.group, EC2_REGION)
    logger.setLevel(logging.INFO)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.type == 'mlm':
        Pipeline = MLMPretrainerPipeline
    elif args.type == 'nsp':
        Pipeline = NSPPretrainerPipeline
    elif args.type == 'scmlm':
        Pipeline = SCMLMPretrainerPipeline
    else:
        Pipeline = MLMPretrainerPipeline

    Pipeline(logger, args.out_dir).e2e(
        in_model_dir=args.in_model_dir,
        ckpt_id=args.model_id,
        model_id=args.model_id,
        ckpt_options=tf.train.CheckpointOptions(experimental_io_device='/job:localhost'),
        training_preprocessing=shorten_ds)
