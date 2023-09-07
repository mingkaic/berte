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
    "training_batches",
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
        self.training_batches = args["training_batches"]

        self.training_loss = args["training_loss"]

    def run_epoch(self, logger=None, nan_reporter=None, report_metric=None):
        """ run a single epoch """
        if logger is None:
            logger = telemetry.EmptyLogger()
        if report_metric is None:
            report_metric = telemetry.get_logger_metric_reporter(logger)

        self.training_loss.reset_states()
        for i, windows in enumerate(self.training_batches):
            debug_info, err_code = self.args["training_cb"](windows)
            if err_code != 0:
                if err_code == NAN_LOSS_ERR_CODE:
                    if nan_reporter is not None:
                        nan_reporter(debug_info)
                    logger.error('batch %d produced nan loss! skipping...', i)
                    # nan is fatal
                    return

            if i % 50 == 0:
                report_metric(
                        Batch=i,
                        Loss=float(self.training_loss.result().numpy()))
