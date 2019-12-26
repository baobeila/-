#!/usr/bin/env python
# encoding: utf-8
"""
@version: JetBrains PyCharm 2017.3.2 x64
@author: baobeila
@contact: baibei891@gmail.com
@software: PyCharm
@file: config.py
@time: 2019/12/20 16:12
"""
"""
Configuration Part.
"""
import os


# path and dataset parameter

# DATA_PATH = './data/tfrecord/train.tfrecords'
DATA_PATH = './data'
OUTPUT_DIR = os.path.join(DATA_PATH, 'output')
#TODO 集成tfrecord文件的生成，生成缓存文件
CACHE_PATH = os.path.join(DATA_PATH, 'cache')
WEIGHTS_FILE  = 'bvlc_alexnet .npy'
CLASSES = ['cat', 'dog']
FLIPPED = True
# model parameter
IMAGE_SIZE = 224
ALPHA = 0.1
#Dropout probability.
KEEP_PROB = 0.5
#List of names of the layer, that get trained from scratch
SKIP_LAYER = ['fc8', 'fc7', 'fc6']
# solver parameter
GPU = '0'
BATCH_SIZE = 32
MAX_ITER = 5000
LEARNING_RATE = 0.0005
DECAY_STEPS = 2000
DECAY_RATE = 0.96
STAIRCASE = True
SUMMARY_ITER = 1#多少次写入日志文件
SAVE_ITER = 2500 #多少次保存模型文件

#
# test parameter
#
THRESHOLD = 0.1




if __name__ == '__main__':
    pass