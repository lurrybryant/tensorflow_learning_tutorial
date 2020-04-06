# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def hello_tf():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    a = tf.constant(10)
    b = tf.constant(32)
    print(sess.run([a + b, hello]))
    sess.close()

def math_op():
    a = tf.constant(3)  # 定义常量3
    b = tf.constant(4)  # 定义常量4

    add = tf.add(a, b)
    mul = tf.multiply(a, b)

    with tf.Session() as sess:  # 建立session
        print("相加: %i" % sess.run(a + b))
        print("相乘: %i" % sess.run(a * b))
        # Run every operation with variable input
        print("相加: %i" % sess.run(add, feed_dict={a: 3, b: 4}))
        print("相乘: %i" % sess.run(mul, feed_dict={a: 3, b: 4}))


def linear_model():
    # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
    x_data = np.float32(np.random.rand(2, 100))  # 随机输入
    y_data = np.dot([0.100, 0.200], x_data) + 0.300

    # 构造一个线性模型
    #
    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    # 最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化变量
    init = tf.initialize_all_variables()

    # 启动图 (graph)
    sess = tf.Session()
    sess.run(init)

    # 拟合平面
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))

    # 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
    sess.close()


def device_demo():
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            matrix1 = tf.constant([[3., 3.]])
            matrix2 = tf.constant([[2.], [2.]])
            product = tf.matmul(matrix1, matrix2)
            print(sess.run(product))


def count_demo():
    # 创建一个变量, 初始化为标量 0.
    state = tf.Variable(0, name="counter")

    # 创建一个 op, 其作用是使 state 增加 1

    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    # 启动图后, 变量必须先经过`初始化` (init) op 初始化,
    # 首先必须增加一个`初始化` op 到图中.
    init_op = tf.initialize_all_variables()

    # 启动图, 运行 op
    with tf.Session() as sess:
        # 运行 'init' op
        sess.run(init_op)
        # 打印 'state' 的初始值
        print(sess.run(state))
        # 运行 op, 更新 'state', 并打印 'state'
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))


def feed_demo():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))


def random_demo():
    # Create a tensor of shape [2, 3] consisting of random normal values, with mean
    # -1 and standard deviation 4.
    norm = tf.random_normal([2, 3], mean=-1, stddev=4)

    # Shuffle the first dimension of a tensor
    c = tf.constant([[1, 2], [3, 4], [5, 6]])
    shuff = tf.random_shuffle(c)

    # Each time we run these ops, different results are generated
    sess = tf.Session()
    print(sess.run(norm), sess.run(norm))

    # Set an op-level seed to generate repeatable sequences across sessions.
    norm = tf.random_normal([2, 3], seed=1234)
    sess = tf.Session()
    print(sess.run(norm))
    print(sess.run(norm))
    sess = tf.Session()
    print(sess.run(norm))
    print(sess.run(norm))


if __name__ == "__main__":
    # hello_tf()
    # linear_model()
    # device_demo()
    # count_demo()
    # feed_demo()
    # random_demo()
    math_op()

