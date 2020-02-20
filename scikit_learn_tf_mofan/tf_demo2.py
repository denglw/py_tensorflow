#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 11:41
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_demo2.py
# @desc: 莫烦视频示例  模拟线性训练参数模型（权重斜率）

import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# create tensorflow sructure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# create tensorflow sructure end

with tf.Session() as sess:
    init.run()
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step,sess.run(Weights),sess.run(biases))
