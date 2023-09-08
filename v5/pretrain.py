#!/usr/bin/env python3
"""
This module calls pretrainers
"""
# standard packages
import subprocess
import os

import shutil

CACHE_DIR = 'cache_dir'

class TrainingMethod:
    """
    TrainingMethod runs a pretraining type (e.g.: mlm, nsp, scmlm)
    """
    def __init__(self, method_type, metric_name='berte'):
        self.method_type = method_type
        self.metric_name = metric_name

    def run(self, cmds, args):
        """ run runs cmd for the pretraining type using the specified metric_name """
        _, err = subprocess.Popen(cmds + (
                '--metric_name', self.metric_name,
                self.method_type,
            ) + args, stdout=subprocess.PIPE).communicate()
        if err:
            print(err)

class PretrainRunner:
    """
    PretrainerRunner allows running multiple methods in series over a number of epochs
    """
    def __init__(self, env, group_id, model_id,
            training_methods=None):
        if training_methods is None:
            training_methods = [
                TrainingMethod('mlm', metric_name='berte'),
                TrainingMethod('nsp', metric_name='berte_nsp'),
            ]

        self.args = ('python3', 'pretrain_{}.py'.format(env),
                '--group', group_id,
                '--model_id', model_id)
        self.group_id = group_id
        self.model_id = model_id
        self.training_methods = training_methods

    def sequence(self, nepochs, in_model_dir, out_dir):
        """
        sequence runs the trainers for nepochs
        """
        cache_path = os.path.join(self.group_id, CACHE_DIR)
        if os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        for epoch in range(nepochs):
            for i, trainer in enumerate(self.training_methods):
                next_path = os.path.join(cache_path, 'epoch{}_interm{}'.format(epoch, i))
                trainer.run(self.args, (in_model_dir, next_path))
                in_model_dir = os.path.join(next_path, self.model_id)

        shutil.copytree(next_path, out_dir, dirs_exist_ok=True)
        shutil.rmtree(cache_path)
