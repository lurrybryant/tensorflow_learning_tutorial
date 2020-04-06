# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 05:05:33 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
import tensorflow as tf


def demo01():
    tf.reset_default_graph()

    var1 = tf.Variable(1.0, name='firstvar')
    print("var1:", var1.name)
    var1 = tf.Variable(2.0, name='firstvar')
    print("var1:", var1.name)
    var2 = tf.Variable(3.0)
    print("var2:", var2.name)
    var2 = tf.Variable(4.0)
    print("var1:", var2.name)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("var1=", var1.eval())
        print("var2=", var2.eval())

    get_var1 = tf.get_variable("firstvar", [1], initializer=tf.constant_initializer(0.3))
    print("get_var1:", get_var1.name)

    # get_var1 = tf.get_variable("firstvar",[1], initializer=tf.constant_initializer(0.4))
    # print ("get_var1:",get_var1.name)

    get_var1 = tf.get_variable("firstvar1", [1], initializer=tf.constant_initializer(0.4))
    print("get_var1:", get_var1.name)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("get_var1=", get_var1.eval())


def demo02():
    tf.reset_default_graph()

    # var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    # var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)

    with tf.variable_scope("test1", ):
        var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

    print("var1:", var1.name)
    print("var2:", var2.name)


def demo03():
    tf.reset_default_graph()

    # var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    # var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)

    with tf.variable_scope("test1", ):
        var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

        with tf.variable_scope("test2"):
            var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

    print("var1:", var1.name)
    print("var2:", var2.name)

    with tf.variable_scope("test1", reuse=True):
        var3 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
        with tf.variable_scope("test2"):
            var4 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

    print("var3:", var3.name)
    print("var4:", var4.name)


def demo04():
    with tf.variable_scope("test1", initializer=tf.constant_initializer(0.4)):
        var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

        with tf.variable_scope("test2"):
            var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
            var3 = tf.get_variable("var3", shape=[2], initializer=tf.constant_initializer(0.3))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("var1=", var1.eval())
        print("var2=", var2.eval())
        print("var3=", var3.eval())


def demo05():
    tf.reset_default_graph()

    with tf.variable_scope("scope1") as sp:
        var1 = tf.get_variable("v", [1])

    print("sp:", sp.name)
    print("var1:", var1.name)

    with tf.variable_scope("scope2"):
        var2 = tf.get_variable("v", [1])

        with tf.variable_scope(sp) as sp1:
            var3 = tf.get_variable("v3", [1])

            with tf.variable_scope(""):
                var4 = tf.get_variable("v4", [1])

    print("sp1:", sp1.name)
    print("var2:", var2.name)
    print("var3:", var3.name)
    print("var4:", var4.name)
    with tf.variable_scope("scope"):
        with tf.name_scope("bar"):
            v = tf.get_variable("v", [1])
            x = 1.0 + v
            with tf.name_scope(""):
                y = 1.0 + v
    print("v:", v.name)
    print("x.op:", x.op.name)
    print("y.op:", y.op.name)

if __name__ == "__main__":
    demo05()