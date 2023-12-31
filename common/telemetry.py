#!/usr/bin/env python3
"""
This module includes logging and reporting functions
"""

import logging

import numpy as np

class EmptyLogger:
    """ Mimicks real logger """
    def info(self, fmt, *args, **kwargs):
        """ does nothing """

    def error(self, fmt, *args, **kwargs):
        """ does nothing """

class PrefixAdapter(logging.LoggerAdapter):
    """ Custom logger adapter for adding prefixes """
    def process(self, msg, kwargs):
        """ overwriting logger adapter """
        return "[{}] {}".format(self.extra, msg), kwargs

def get_logger_metric_reporter(logger):
    """ Return metric report function by calling logger.info against the metric kwargs """
    def _report_metric(**kwargs):
        logger.info(' '.join(['{} {}'.format(key, value) for key, value in kwargs.items()]))
    return _report_metric

def detail_reporter(logger):
    """ Return reporting function that detokenize tokens and error to sentences """
    def _reporter(debug_info):
        for debug_key in debug_info:
            debug_val = debug_info[debug_key].numpy()
            hasnan = np.isnan(debug_val).any()
            logger.error('{}:{} (has nan: {})'.format(debug_key, debug_val, hasnan))
    return _reporter
