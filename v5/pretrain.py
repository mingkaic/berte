#!/usr/bin/env python3
"""
This module calls pretrainers
"""
# standard packages
import subprocess
import os

import shutil

CACHE_DIR = 'cache_dir'

class PretrainRunner:
    def __init__(self, env, group_id, model_id,
            mlm_metric=None,
            nsp_metric=None):
        self.args = ('python3', 'pretrain_{}.py'.format(env),
                '--group', group_id,
                '--model_id', model_id)
        self.group_id = group_id
        self.model_id = model_id
        self.mlm_metric = mlm_metric
        self.nsp_metric = nsp_metric

    def run_mlm(self, in_dir, out_dir):
        if self.mlm_metric is not None:
            metric_name = self.mlm_metric
        else:
            metric_name = 'berte'
        self._run(in_dir, out_dir, 'mlm', metric_name)

    def run_nsp(self, in_dir, out_dir):
        if self.nsp_metric is not None:
            metric_name = self.nsp_metric
        else:
            metric_name = 'berte'
        self._run(in_dir, out_dir, 'nsp', metric_name)

    def _run(self, in_dir, out_dir, ptype, metric_name):
        _, err = subprocess.Popen(self.args + (
            '--metric_name', metric_name,
            ptype, in_dir, out_dir), stdout=subprocess.PIPE).communicate()
        if err:
            print(err)

    def sequence(self, nepochs, in_model_dir, out_dir):
        cache_path = os.path.join(self.group_id, CACHE_DIR)
        if os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        for epoch in range(nepochs):
            interm_path = os.path.join(cache_path, 'interm{}'.format(epoch))
            self.run_mlm(in_model_dir, interm_path)

            next_path = os.path.join(cache_path, 'next{}'.format(epoch))
            self.run_nsp(os.path.join(interm_path, self.model_id), next_path)

            in_model_dir = os.path.join(next_path, self.model_id)

        shutil.copytree(next_path, out_dir, dirs_exist_ok=True)
        shutil.rmtree(cache_path)
