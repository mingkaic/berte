#!/usr/bin/env python3
"""
This tests pretrain locally
"""
# standard packages
import shutil
import psutil

from pretrain import PretrainRunner

if __name__ == '__main__':
    ID = 'test'

    process = psutil.Process()
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    PretrainRunner('test_local', ID, ID).sequence(5, '', ID)
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')

    shutil.rmtree(ID) # remove checkpoints
    print(str(process.memory_info().rss / (1024 * 1024 * 1024)) + 'GB')
