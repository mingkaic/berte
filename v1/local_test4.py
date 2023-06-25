#!/usr/bin/env python3
"""
This tests 4_persentence_pretrain_mlm_15p.ipynb locally
"""
# standard packages
import logging

# business logic
from ps_pretrain_mlm_15p import main

def shorten_shards(training_shards):
    """ take 1 from each shard and take the first 3 shads """
    result = dict()
    for training_key in list(training_shards.keys())[:2]:
        result[training_key] = training_shards[training_key].take(1)
    return result

if __name__ == '__main__':
    logging.basicConfig(filename="tmp/4_persentence_pretrain_mlm_15p.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main(logger, 'tmp',
            ckpt_id='test',
            model_id='test',
            training_preprocessing=shorten_shards,
            context_rate_overwrite=2.0)
