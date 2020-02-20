#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/2/19 17:48
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_book_11_learn_rate.py
# @desc:  学习率调整learning rate

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
X = tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y = tf.placeholder(tf.int64,shape=(None),name="x")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X,n_hidden1,activation=tf.nn.relu,name="hidden1")
    hidden2 = tf.layers.dense(hidden1,n_hidden2,activation=tf.nn.relu,name="hidden2")
    logits = tf.layers.dense(hidden2,n_outputs,name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")

with tf.name_scope("train"):     #not shown in the book
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0,trainable=False,name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps,decay_rate) #指数衰减
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9)
    training_op = optimizer.minimize(loss,global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 50

with tf.Session() as sess:
    sess.run(init)
    for epoche in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})

        accuracy_val = accuracy.eval(feed_dict={X:mnist.test.images,y:mnist.test.labels})
        print(epoche,"Test accuracy:",accuracy_val)
    save_path = saver.save(sess,"./my_model_final.ckpt")



