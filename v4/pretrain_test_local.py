#!/usr/bin/env python3
"""
This runs pretrain_mlm or pretrain_nsp on the local machine
"""
# standard packages
import logging
import os.path
import argparse

# business logic
from pretrain_mlm import PretrainerPipeline as MLMPretrainerPipeline
from pretrain_nsp import PretrainerPipeline as NSPPretrainerPipeline

def shorten_ds(training_ds):
    """ take 36 from the dataset """
    return training_ds.take(36)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='pretrain_mlm_local',
                    description='runs pretrain_mlm on the local environment')
    parser.add_argument('type')
    parser.add_argument('in_model_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--group')
    parser.add_argument('--model_id')
    parser.add_argument('--metric_name', default='berte')
    args = parser.parse_args()

    if not os.path.exists(args.group):
        os.makedirs(args.group, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.group, "pretrain.log"),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if args.type == 'mlm':
        Pipeline = MLMPretrainerPipeline
    elif args.type == 'nsp':
        Pipeline = NSPPretrainerPipeline
    else:
        Pipeline = MLMPretrainerPipeline

    Pipeline(logger, args.out_dir).e2e(
        in_model_dir=args.in_model_dir,
        ckpt_id=args.model_id,
        model_id=args.model_id,
        training_preprocessing=shorten_ds)
