#!/usr/bin/env python3
"""
This module defines a common builder when passing a lot of arguments to some method or object
instantiation.
"""

import functools

class Builder:
    """ build adds args keys as class methods """
    def __init__(self, args_keys=None, default_values=None):
        if default_values is not None:
            self.args = dict(default_values)
        else:
            self.args = dict()

        if args_keys is not None:
            for key in args_keys:
                setattr(self.__class__, key, functools.partial(self.add, key=key))

    def add(self, value, key):
        """ add custom key value """
        self.args[key] = value
        return self

    def build(self):
        """ return built arguments """
        return self.args
