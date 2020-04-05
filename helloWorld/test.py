# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

x = tf.constant([[1, 4],
                [2,3],
                 [2,3]])
# m = x[:,0]
# t = tf.reshape(m, (-1,1))
y = tf.constant([2, 2, 2])
# y = tf.reshape(y, (-1,1))
h = tf.stack([x[:,0], y])
z = tf.transpose(h)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

print(sess.run([h, z]))


