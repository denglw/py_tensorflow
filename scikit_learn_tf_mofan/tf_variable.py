#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 16:49
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_variable.py
# @desc: Variable变量

import tensorflow as tf
import numpy as np

state = tf.Variable(0,name='counter')
print(state.name)
one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

init = tf.initialize_all_variables() # 初始化变量 声明构造
with tf.Session() as sess:
    sess.run(init) # 初始化变量 执行
    for step in range(3):
        sess.run(update)
        temp = sess.run(state)
        print(temp)

