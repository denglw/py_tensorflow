#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2019/9/6 9:43
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_loss_function.py
# @desc: 损失函数loss function ：衡量模型模型预测的好坏

print('损失函数')
'''1、回归损失'''
# 1、 均方误差（MSE）度量的是预测值和实际观测值间差的平方的均值。
import numpy as np
y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])
def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val
print("d is: " + str(["%.8f" % elem for elem in y_hat]))
print("p is: " + str(["%.8f" % elem for elem in y_true]))
rmse_val = rmse(y_hat, y_true)
print("rms error is: " + str(rmse_val))

# 2、平均绝对误差（MAE）度量的是预测值和实际观测值之间绝对差之和的平均值。 L1损失
import numpy as np
y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])

print("d is: " + str(["%.8f" % elem for elem in y_hat]))
print("p is: " + str(["%.8f" % elem for elem in y_true]))

def mae(predictions, targets):
    differences = predictions - targets
    absolute_differences = np.absolute(differences)
    mean_absolute_differences = absolute_differences.mean()
    return mean_absolute_differences

mae_val = mae(y_hat, y_true)
print ("mae error is: " + str(mae_val))


# 3、平均偏差误差（mean bias error）
'''它与 MAE 相似，唯一的区别是这个函数没有用绝对值。用这个函数需要注意的一点是，正负误差可以互相抵消。
尽管在实际应用中没那么准确，但它可以确定模型存在正偏差还是负偏差。'''

'''2、分类损失'''
# 1、Hinge Loss/多分类 SVM 损失
'''
简言之，在一定的安全间隔内（通常是 1），正确类别的分数应高于所有错误类别的分数之和。
因此 hinge loss 常用于最大间隔分类（maximum-margin classification），最常用的是支持向量机。
尽管不可微，但它是一个凸函数，因此可以轻而易举地使用机器学习领域中常用的凸优化器。
'''

# 2、交叉熵损失/负对数似然：
'''
这是分类问题中最常见的设置。随着预测概率偏离实际标签，交叉熵损失会逐渐增加。
注意，当实际标签为 1(y(i)=1) 时，函数的后半部分消失，而当实际标签是为 0(y(i=0)) 时，函数的前半部分消失。
简言之，我们只是把对真实值类别的实际预测概率的对数相乘。
还有重要的一点是，交叉熵损失会重重惩罚那些置信度高但是错误的预测值。
'''
import numpy as np
predictions = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.96]])
targets = np.array([[0,0,0,1],
                   [0,0,0,1]])
def cross_entropy(predictions, targets, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5)))/N
    return ce_loss
cross_entropy_loss = cross_entropy(predictions, targets)
print ("Cross entropy loss is: " + str(cross_entropy_loss))



