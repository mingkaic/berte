#!/usr/bin/env python3
"""
This module includes pretraining each epoch
"""

import functools

from export.mlm import NAN_LOSS_ERR_CODE

import common.telemetry as telemetry

EPOCH_PRETRAINER_ARGS = [
    "training_settings",
    "training_loss",
    "training_accuracy",
    "training_shards",
    "training_cb",
    "ckpt_save_cb",
]

class EpochPretrainerInitBuilder:
    """ build pretrainer init arguments """
    def __init__(self):
        self.arg = dict()
        for key in EPOCH_PRETRAINER_ARGS:
            setattr(self.__class__, key, functools.partial(self.add_arg, key=key))

    def add_arg(self, value, key):
        """ add custom arg """
        self.arg[key] = value
        return self

    def build(self):
        """ return built arg """
        return self.arg

class EpochPretrainer:
    """ Reusable pretrainer for running training epochs """
    def __init__(self, args):

        self.args = args
        self.training_shards = args["training_shards"]
        self.training_keys = list(self.training_shards.keys())

        self.training_loss = args["training_loss"]
        self.training_accuracy = args["training_accuracy"]

    def run_epoch(self,
            skip_shards=None,
            logger=None,
            nan_reporter=None,
            report_metric=None):
        """ run a single epoch """
        if logger is None:
            logger = telemetry.EmptyLogger()
        if report_metric is None:
            report_metric = telemetry.get_logger_metric_reporter(logger)

        self.training_loss.reset_states()
        self.training_accuracy.reset_states()
        batch = 0
        for (shard, training_key) in enumerate(self.training_keys):
            logger.info('Training shard %s', training_key)
            if skip_shards is not None and training_key in skip_shards:
                skip_it = self.training_shards[training_key].cardinality()
                logger.info('skipped %s: fastforward %d iterations', training_key, skip_it)
                batch += skip_it
                skip_shards.remove(training_key)
                continue

            for tokens, lengths in self.training_shards[training_key]:
                debug_info, err_code = self.args["training_cb"](tokens, lengths)
                if err_code != 0:
                    if err_code == NAN_LOSS_ERR_CODE:
                        if nan_reporter is not None:
                            nan_reporter(debug_info)
                        logger.error('batch %d produced nan loss! skipping...', batch)
                        # nan is fatal
                        return

                if batch % 50 == 0:
                    report_metric(
                            Batch=batch,
                            Loss=float(self.training_loss.result().numpy()),
                            Accuracy=float(self.training_accuracy.result().numpy()))

                batch += 1

            if (shard + 1) % self.args['training_settings']['shards_per_save'] == 0:
                logger.info('Saving checkpoint for shard %d at %s', shard+1,
                        self.args['ckpt_save_cb'](None))
