#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2019/12/20 17:23
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_book_11_restore.py
# @desc: 机器学习scikit-learn和tensorflow 第11章  复用深度神经网络

from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 1、构建原始模型 construct the original model
if __name__ == '__main__':

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 50
    n_hidden3 = 50
    n_hidden4 = 50
    n_hidden5 = 50
    n_outputs = 10

    mnist = input_data.read_data_sets("/tmp/data/")

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
        hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
        logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    learning_rate = 0.01
    threshold = 1.0

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    # 梯度裁剪
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
                  for grad, var in grads_and_vars]
    training_op = optimizer.apply_gradients(capped_gvs)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epoches = 20
    batch_size = 200

    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")

        for epoch in range(n_epoches):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                    y: mnist.test.labels})
            print(epoch, "Test accuracy:", accuracy_val)

        save_path = saver.save(sess, "./my_new_model_final.ckpt")





