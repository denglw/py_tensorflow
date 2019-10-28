#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 17:27
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_placeholder.py
# @desc:Place holder占位符

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)
with tf.Session() as sess:
    result = sess.run(output,feed_dict={input1:[7.],input2:[2.]}) # 占位符赋值 字典格式
    print(result)
