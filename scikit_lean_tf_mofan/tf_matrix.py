#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 16:09
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_matrix.py
# @desc: 矩阵相乘

import tensorflow as tf
import numpy as np

'''
1、当矩阵A的列数（column）等于矩阵B的行数（row）时，A与B可以相乘。
2、矩阵C的行数等于矩阵A的行数，C的列数等于B的列数。
3、乘积C的第m行第n列的元素等于矩阵A的第m行的元素与矩阵B的第n列对应元素乘积之和。
'''
matrix1 = tf.constant([[3,3]]) # 1行2列
matrix2 = tf.constant([[2],[2]]) # 2行1列
product = tf.matmul(matrix1,matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

