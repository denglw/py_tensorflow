# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:45:13 2019

@author: Administrator
"""

import tensorflow as tf
# tensorflow Version : 1.14.0
# numpy install : pip install numpy=1.16.4
print("Version : %s" % tf.__version__)
print("Path : %s" % tf.__path__)

# 简单测试  TensorFlow环境
import tensorflow as tf
sess = tf.Session()
a = tf.constant(10)
b= tf.constant(12)
result = sess.run(a+b)
print(result)

