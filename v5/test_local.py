#!/usr/bin/env python3
"""
This tests pretrain locally
"""
# standard packages
import shutil
import psutil

from common.pretrain import PretrainRunner, TrainingMethod

ID = 'test'
MODEL_DIR = 'intake/berte_pretrain'
MODEL_ID = 'berte_pretrain'

if __name__ == '__main__':
    process = psutil.Process()
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    PretrainRunner('test_local', ID, MODEL_ID,
            { 'group': ID, 'model_id': MODEL_ID },
            training_methods = [
                TrainingMethod('mlm', {'metric_name': 'berte'}),
                TrainingMethod('nsp', {'metric_name': 'berte_nsp'}),
            ]).sequence(5, MODEL_DIR, ID)
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    shutil.rmtree(ID) # remove checkpoints
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')
