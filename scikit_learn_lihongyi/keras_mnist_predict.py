#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/4/12 14:17
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : keras_mnist_predict.py
# @desc:  keras2.0   mnist手写数字   预测
# tensorflow 1.14.0  keras2.3.1
# caffe框架
# https://blog.csdn.net/pcb931126/article/details/81412059
# https://blog.csdn.net/linglian0522/article/details/78346237
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.python.keras.models import load_model

# 数据集
# (_, _), (X_test, y_test) = down_local_data()  # 划分MNIST训练集、测试集
(_, _), (X_test, y_test) = mnist.load_data()  # 划分MNIST训练集、测试集

# 加载模型
mymodel = load_model('mnistmodel.h5')
# 随机数
index = random.randint(0, X_test.shape[0])
x = X_test[index]
y = y_test[index]
# 显示数字
plt.imshow(x,cmap='gray_r')
plt.title("original {}".format(y))
plt.show()

# 预测
x.shape = (1,784)#变成[[]]
# x = x.flatten()[None]  # 也可以用这个
predict = mymodel.predict(x)
predict = np.argmax(predict)#取最大值的位置
print('index', index)
print('original:', y)
print('predicted:', predict)


