#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2020/2/16 0:14
# @Author  : De
# er.py
# @desc: model saver and restore

from __future__ import print_function
import tensorflow as tf
import numpy as np

# 1、Save to file
# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#
# # tf.initialize_all_variables() no long valid from
# # 2017-03-02 if using tensorflow >= 0.12
# if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#     init = tf.initialize_all_variables()
# else:
#     init =tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,"my_net/save_net.ckpt")
#     print("Save to path: ", save_path)


#########################################################
# 2、restore variables
# redefine the same shape and same type for your variables
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"my_net/save_net.ckpt")
    print("weights: " , sess.run(W))
    print("biases: ",sess.run(b))



