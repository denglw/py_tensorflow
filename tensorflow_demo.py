#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2019/8/12 17:14
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tensorflow_demo.py
# @desc: tensorflow环境测试：创建一个计算图并在会话中执行

import tensorflow as tf
x = tf.Variable(3, name="x") #定义变量（节点）
y = tf.Variable(4, name="y")
f = x*x*y + y + 2  # 创建一个计算图
sess = tf.Session()  # 创建一个session
sess.run(x.initializer)  # 初始化变量
sess.run(y.initializer)
result = sess.run(f)  # 运行求值
print(result)
sess.close() #释放空间

#  每次都重复sess.run（）看起来有些笨拙，好在有更好的方式：
#  在with块中，会有一个默认会话。
# 调用x.initializer.run等价于调用 tf.get_default_session（）.run（x.initializer）
# 还可使会话在块中的代码执行结束后自动关闭

# a better way
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval() #即evaluate，求解f的值
print(result)







