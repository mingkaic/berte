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

def tokens_reporter(logger, tokenizer):
    """ Return reporting function that detokenize tokens and error to sentences """
    def _reporter(tokens):
        if isinstance(tokens, tf.Tensor):
            return

        sentences = tokenizer.detokenizer(tokens).numpy()
        for sentence in sentences:
            logger.error(sentence)
    return _reporter
