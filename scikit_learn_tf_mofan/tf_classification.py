#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2020/2/9 15:01
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_classification.py
# @desc: 分类归类学习 tf.nn.softmax

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# renumber 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation_function=None):
    # add one more layer and return ouputs in this layer
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.arg_max(y_pre,1),tf.arg_max(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# define placeholder for inputs network
xs = tf.placeholder(tf.float32,[None,784]) # 28*28
ys = tf.placeholder(tf.float32,[None,10])

# add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax) # softmax classification

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

