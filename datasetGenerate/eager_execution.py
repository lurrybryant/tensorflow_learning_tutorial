# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import os
import tensorflow as tf

from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()
print("TensorFlow 版本: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

