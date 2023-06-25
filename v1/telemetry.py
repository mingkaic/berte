#!/usr/bin/env python3
"""
This module includes logging and reporting functions
"""

import logging

import tensorflow as tf

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

def detail_reporter(logger, tokenizer):
    """ Return reporting function that detokenize tokens and error to sentences """
    def _reporter(tokens, debug_info):
        if isinstance(tokens, tf.Tensor):
            sentences = tokenizer.detokenizer(tokens).numpy()
            logger.error('sentences:{}'.format(','.join([
                '"' + sentence + '"' for sentence in sentences])))
        for debug_key in debug_info:
            logger.error('{}:{}'.format(debug_key, debug_info[debug_key].numpy()))
    return _reporter
