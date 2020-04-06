#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 22:05
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : keras_mnist.py
# @desc:  keras2.0   mnist手写数字识别
# tensorflow 1.14.0  keras2.3.1
# https://blog.csdn.net/weixin_41122036/article/details/89003521

import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Dense,Dropout
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.models import Sequential

def down_internet_data():
    # 网络下载失败
    (x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')  # 网络下载数据
    return (x_train, y_train), (x_test, y_test)

def down_local_data():
    # 读取本地数据
    data_path = 'datasets/mnist.npz'
    cwd_dir = os.path.dirname(os.getcwd()) #获取文件目录
    path = os.path.join(cwd_dir,data_path)
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train),(x_test, y_test)

def load_data():
    # (x_train, y_train), (x_test, y_test) = down_internet_data()
    (x_train, y_train), (x_test, y_test) = down_local_data()
    # (x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')
    number = 10000
    x_train = x_train[0: number]
    y_train = y_train[0: number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    # x_test = np.random.normal(x_test)
    x_train = x_train / 255
    x_test = x_test / 255
    # x_test = np.random.normal(x_test)
    return (x_train, y_train), (x_test, y_test)

# 下载
(x_train, y_train), (x_test, y_test) = load_data()

# 引入模型 训练&测试 输出测试结果
model = Sequential()
model.add(Dense(input_dim=28*28, units=633,activation='sigmoid'))
# model.add(Dropout(0.5)) # dropout
model.add(Dense(units=633,activation='sigmoid'))
# model.add(Dropout(0.5)) # dropout
model.add(Dense(units=633,activation='sigmoid'))
# for i in range(10):
#     model.add(Dense(units=633, activation='sigmoid'))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='mse',optimizer=SGD(lr=0.1), metrics = ['accuracy'])
# model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1), metrics = ['accuracy'])
# model.compile(loss='categorical_crossentropy',optimizer=Adam, metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)

score = model.evaluate(x_train,y_train)
print('\n Train Loss:',score[0])
print('\n Train Acc:',score[1])
score = model.evaluate(x_test,y_test)
print('\n Test Loss:',score[0])
print('\n Test Acc:',score[1])





