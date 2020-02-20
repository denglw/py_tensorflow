#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2019/12/20 17:47
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_book_11_restore_part.py
# @desc: 机器学习scikit-learn和tensorflow 第11章  复用部分，深度神经网络

from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 只需要重新使用原始模型的一部分（就像我们将要讨论的那样）。
# 一个简单的解决方案是将Saver配置为仅恢复原始模型中的一部分变量。
# 例如，下面的代码只恢复隐藏的层1,2和3：

if __name__ == '__main__':

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300 # reused
    n_hidden2 = 50  # reused
    n_hidden3 = 50  # reused
    n_hidden4 = 20  # new!
    n_outputs = 10  # new!

    mnist = input_data.read_data_sets("/tmp/data/")

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")       # reused
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2") # reused
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3") # reused
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4") # new!
        logits = tf.layers.dense(hidden4, n_outputs, name="outputs")                         # new!

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    learning_rate = 0.01
    threshold = 1.0

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    # build new model with the same definition as before for hidden layers 1-3
    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope="hidden[123]") # regular expression
    reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
    restore_saver = tf.train.Saver(reuse_vars_dict) # to restore layers 1-3

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epoches = 20
    batch_size = 200

    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "./my_model_final.ckpt")

        for epoch in range(n_epoches):                                      # not shown in the book
            for iteration in range(mnist.train.num_examples // batch_size): # not shown
                X_batch, y_batch = mnist.train.next_batch(batch_size)      # not shown
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})  # not shown
            accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,  # not shown
                                                    y: mnist.test.labels}) # not shown
            print(epoch, "Test accuracy:", accuracy_val)                   # not shown

        save_path = saver.save(sess, "./my_new_model_final.ckpt")
