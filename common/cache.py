#!/usr/bin/env python3
"""
This module sets up argument caching
"""

import yaml

def cache_values(cache_file, generators, *args, **kwargs):
    """
    read from cache_file if it exist and populate missing arguments from generators with args
    """
    cached_args = dict()
    try:
        with open(cache_file, "r") as file:
            args = yaml.safe_load(file.read())
            if args is not None:
                cached_args = args
    except FileNotFoundError:
        pass

    for key in generators:
        if key not in cached_args:
            cached_args[key] = generators[key](*args, **kwargs)

    with open(cache_file, "w") as file:
        file.write(yaml.dump(cached_args))

    return cached_args
