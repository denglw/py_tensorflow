#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2019/8/13 14:38
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tensorflow_test.py
# @desc:


import tensorflow as tf
import numpy as np
import datetime

starttime = datetime.datetime.now()
# 制造数据
train_X = np.linspace(-1,1,100)
train_Y = 2*train_X+np.random.randn(*train_X.shape)*0.3
# y =2x+b


# 训练模型

train_epochs = 200
display_step =4 #展示模型参数
def moving_average(a,w=10):
    if len(a)<w:
        return a[:]
    return [val if i<w else sum(a[(i-w):i])/w for i,val in enumerate(a)]

# config = tf.ConfigProto(log_device_placement = True,allow_soft_placement=True)
# config.gpu_options.allow_growth = True
with tf.Session() as sess:
    # with tf.device("/gpu:0"):
        # 创建模型
        X = tf.placeholder("float")
        Y = tf.placeholder("float")
        W = tf.Variable(tf.random_normal([1]), name="weight")
        b = tf.Variable(tf.zeros([1]), name='bias')
        z = tf.multiply(X, W) + b

        cost = tf.reduce_mean(tf.square(Y - z))  # 损失函数
        learning_rate = 0.01  # 学习率，越小精度越高，速度越慢
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # 梯度下降算法

        pltdata = {'batchsize': [], "loss": []}  # 存放批次值和损失值
        sess.run(tf.global_variables_initializer())

        # 向模型填充数据
        for epoch in range(train_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})
                if epoch % display_step == 0:
                    loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                    print("Epoch", epoch + 1, "cost", loss, "W=", sess.run(W), "b=", sess.run(b))
                    if not (loss == 'NA'):
                        pltdata['batchsize'].append(epoch)
                        pltdata['loss'].append(loss)

        print('完成了。')
        print('cost=',sess.run(cost,feed_dict={X:train_X,Y:train_Y}),'W=',sess.run(W),'b=',sess.run(b))
        # plt.show()
        #使用模型预测
        print(sess.run(z,feed_dict={X:0.2}))
endtime = datetime.datetime.now()

print((endtime - starttime).seconds)
