#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
# @Time    : 2020/2/20 15:55
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_book_11_dropout.py
# @desc: dropout 解决 overfitting 过拟合问题

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
X = tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y = tf.placeholder(tf.int64,shape=(None),name="x")

training = tf.placeholder_with_default(False, shape=(), name='training')
learning_rate = 0.01  # 0.1和0.01结果区别
dropout_rate = 0.5  # == 1 - keep_prob
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,name="hidden1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,name="hidden2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={training: True, X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")

