#!/usr/bin/env python3
"""
This module calls pretrainers
"""
# standard packages
import functools
import subprocess
import os

import shutil

CACHE_DIR = 'cache_dir'

def parse_flag_args(flags):
    flagset = [('--'+key, flags[key]) for key in flags]
    return functools.reduce(lambda ltuple, rtuple: ltuple + rtuple, flagset)

class TrainingMethod:
    """
    TrainingMethod runs a pretraining type (e.g.: mlm, nsp, scmlm)
    """
    def __init__(self, method_type, flags):
        self.method_type = method_type
        self.flags = flags

    def run(self, cmds, args):
        """ run runs cmd for the pretraining type using the specified metric_name """
        _, err = subprocess.Popen(cmds + parse_flag_args(self.flags) +\
                (self.method_type,) + args,
                stdout=subprocess.PIPE).communicate()
        return err

class PretrainRunner:
    """
    PretrainerRunner allows running multiple methods in series over a number of epochs
    """
    def __init__(self, env, group_id, model_id, flags,
            training_methods=None):
        if training_methods is None:
            self.training_methods = [
                TrainingMethod('mlm', {'metric_name': 'berte'}),
                TrainingMethod('nsp', {'metric_name': 'berte_nsp'}),
            ]
        else:
            self.training_methods = training_methods

        self.args = ('python3', 'pretrain_{}.py'.format(env)) + parse_flag_args(flags)
        self.group_id = group_id
        self.model_id = model_id

    def sequence(self, nepochs, in_model_dir, out_dir):
        """
        sequence runs the trainers for nepochs
        """
        cache_path = os.path.join(self.group_id, CACHE_DIR)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        for epoch in range(nepochs):
            for i, trainer in enumerate(self.training_methods):
                next_path = os.path.join(cache_path, 'epoch{}_interm{}'.format(epoch, i))
                err = trainer.run(self.args, (in_model_dir, next_path))
                if err:
                    return err
                in_model_dir = os.path.join(next_path, self.model_id)

        shutil.copytree(next_path, out_dir, dirs_exist_ok=True)
        shutil.rmtree(cache_path)
        return None
