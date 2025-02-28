#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
# @Time    : 2020/2/4 14:43
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_add_layer.py
# @desc: 莫烦视频 定义添加神经层函数add_layer()，以及训练过程可视化 TensorBoard

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
predition = add_layer(l1,10,1,activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
# writer = tf.train.SummaryWriter("logs/",sess.graph)  # 训练图形TensorBoard
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion() #继续执行
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data})) #打印损失函数
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predition_value = sess.run(predition,feed_dict={xs:x_data})
        lines = ax.plot(x_data,predition_value,'-r',lw=5) #机器学习展示x,y的变化过程
        plt.pause(0.1) #暂停0.1秒

'''
Optimizer优化器
SGD
Momentum
NAG
Adagrad
Adadelta
Rmsprop
'''

