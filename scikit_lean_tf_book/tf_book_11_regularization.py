#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/2/20 14:54
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_book_11_regularization.py
# @desc: 内核正则化参数（正则化函数有l1_regularizer()，l2_regularizer()，l1_l2_regularizer()）


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import functools

n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
X = tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y = tf.placeholder(tf.int64,shape=(None),name="x")

scale = 0.001

my_dense_layer = functools.partial(tf.layers.dense, activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))  # kernel_regularizer内核正则化参数  L1

with tf.name_scope("dnn"):
    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
    logits = my_dense_layer(hidden2, n_outputs, activation=None, name="outputs")


# TensorFlow 会自动将这些节点添加到包含所有正则化损失的特殊集合中。
# 您只需要将这些正则化损失添加到您的整体损失中
with tf.name_scope("loss"):                                     # not shown in the book
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # not shown
        labels=y, logits=logits)                                # not shown
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")   # not shown
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

