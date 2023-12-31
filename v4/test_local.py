#!/usr/bin/env python3
"""
This tests pretrain locally
"""
# standard packages
import shutil
import psutil

from common.pretrain import PretrainRunner

if __name__ == '__main__':
    ID = 'test'
    MODEL_DIR = 'intake/berte_pretrain'

    process = psutil.Process()
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    PretrainRunner('test_local', ID, ID,
            {
                'group': ID,
                'model_id': ID,
            }).sequence(5, MODEL_DIR, ID)
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    shutil.rmtree(ID) # remove checkpoints
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')
