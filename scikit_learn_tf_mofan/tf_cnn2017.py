#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2020/2/24 12:37
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_cnn2017.py
# @desc: 卷积神经网络(matplotlib代码有个异常)

"""
W:图像宽，H:图像高，D:图像深度（通道数）
F：卷积核宽高，N:卷积核（过滤器）个数
S:步长，P:用零填充个数
卷积层输出计算公式：（代表向下取整）
Width=(W-F+2P)/S+1
Height=(H-F+2P)/S+1
D = N
池化层输出计算公司：（代表向上取整）
W=(W-F)/S+1
H=(H-F)/S+1
D = N
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as plt

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001 # learning rate

mnist = input_data.read_data_sets('./mnist',one_hot=True) # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape) # (55000, 28 * 28)
print(mnist.train.labels.shape)  # (55000, 10)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()

tf_x = tf.placeholder(tf.float32,[None,28*28]) / 255.
image = tf.reshape(tf_x,[-1,28,28,1])
tf_y = tf.placeholder(tf.int32,[None,10])

# CNN
# shape (28, 28, 1)
conv1 = tf.layers.conv2d(inputs=image,filters=16,kernel_size=5,strides=1,padding='same',activation=tf.nn.relu)
# -> (14, 14, 16)
pool1 = tf.layers.max_pooling2d(conv1,pool_size=2,strides=2)
# -> (14, 14, 32)
conv2 = tf.layers.conv2d(pool1,32,5,1,'same',activation=tf.nn.relu)
# -> (7, 7, 32)
pool2 = tf.layers.max_pooling2d(conv2,2,2)
# -> (7*7*32, )
flat = tf.reshape(pool2,[-1,7*7*32])
output = tf.layers.dense(flat,10) # output layer

# compute cost
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# return (acc, update_op), and create 2 local variables
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y,aixs=1),predictions=tf.argmax(output,axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op) # initialize var in graph

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try:
    from sklearn.manifold import TSNE
    HAS_SK = True
except:
    HAS_SK = False
    print('\nPlease install sklearn for layer visualization\n')

def plot_with_labels(lowDWeights,labels):
    plt.cla()
    X,Y = lowDWeights[:,0], lowDWeights[:,1]
    for x,y,s in zip(X,Y,labels):
        c = cm.rainbow(int(255*s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.xlim(X.min(),X.max())
        plt.ylim(Y.min(),Y.max())
        plt.title('Visualize last layer')
        plt.show()
        plt.pause(0.01)

plt.ion()
for step in range(600):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

        if HAS_SK:
            # Visualization of trained flatten layer (T-SNE)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
            low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
            labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')
