#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 11:27
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_book_09.py
# @desc: 机器学习scikit-learn和tensorflow 第9章 运行TensorFlow

# 1、线性回归
# 使用tensorflow
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
#获得数据维度，矩阵的行列长度
m, n = housing.data.shape
#np.c_是连接的含义，加了一个全为1的维度
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
#数据量并不大，可以直接用常量节点装载进来，但是之后海量数据无法使用（会用minbatch的方式导入数据）
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
#转置成列向量
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
#使用normal equation的方法求解theta，之前线性模型中有提及
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
#求出权重
with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)


# 使用numpy
X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
# 使用sklearn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))


# 2、手工梯度下降
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
# 使用gradient时需要scale一下
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housing = fetch_california_housing()
m, n = housing.data.shape
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
# 迭代1000次
n_epochs = 1000
learning_rate = 0.01
# 由于使用gradient，写入x的值需要scale一下
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# 使用gradient需要有一个初值
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
# 当前预测的y，x是m*（n+1），theta是（n+1）*1，刚好是y的维度
y_pred = tf.matmul(X, theta, name="predictions")
# 整体误差
error = y_pred - y
# TensorFlow求解均值功能强大，可以指定维数，也可以像下面方法求整体的
mse = tf.reduce_mean(tf.square(error), name="mse")
# 暂时自己写出训练过程，实际可以采用TensorFlow自带的功能更强大的自动求解autodiff方法
gradients = 2/m * tf.matmul(tf.transpose(X), error)
#gradients = tf.gradients(mse, [theta])[0]  #   TensorFlow的autodiff
training_op = tf.assign(theta, theta - learning_rate * gradients)
# 初始化并开始求解
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        # 每运行100次打印一下当前平均误差
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()


# 3、给训练算法提供数据
import tensorflow as tf
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
print(B_val_1)
print(B_val_2)



# 4、通过定义min_batch来分批次随机抽取指定数量的数据
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
housing = fetch_california_housing()
m, n = housing.data.shape
# 批次100
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
#有放回的随机抽取数据
def fetch_batch(epoch, batch_index, batch_size):
    #定义一个随机种子
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

# 迭代1000次
n_epochs = 1000
learning_rate = 0.01
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
# 由于使用gradient，写入x的值需要scale一下
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# 使用gradient需要有一个初值
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
# 当前预测的y，x是m*（n+1），theta是（n+1）*1，刚好是y的维度
y_pred = tf.matmul(X, theta, name="predictions")
# 整体误差
error = y_pred - y
# TensorFlow求解均值功能强大，可以指定维数，也可以像下面方法求整体的
mse = tf.reduce_mean(tf.square(error), name="mse")
# 暂时自己写出训练过程，实际可以采用TensorFlow自带的功能更强大的自动求解autodiff方法
#gradients = 2/m * tf.matmul(tf.transpose(X), error)
gradients = tf.gradients(mse, [theta])[0]  #   TensorFlow的autodiff自动求解
training_op = tf.assign(theta, theta - learning_rate * gradients)
# 初始化并开始求解
init = tf.global_variables_initializer()
#开始运行
with tf.Session() as sess:
    sess.run(init)
#每次都抽取新的数据做训练
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#最终结果
    best_theta = theta.eval()
    print(best_theta)



# 5、模型的保存和恢复
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行,{}列".format(m,n))
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000  # not shown in the book
learning_rate = 0.01  # not shown

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")  # not shown
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")  # not shown
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")  # not shown
error = y_pred - y  # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")  # not shown
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # not shown
training_op = optimizer.minimize(mse)  # not shown

init = tf.global_variables_initializer()
saver = tf.train.Saver() #训练模型保存

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())  # not shown
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)

    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt") #找到tmp文件夹就找到文件了



# 6、使用 TensorBoard 展现图形和训练曲线
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行,{}列".format(m,n))
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = r"D://tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess:                                                        # not shown in the book
    sess.run(init)                                                                # not shown

    for epoch in range(n_epochs):                                                 # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
file_writer.close()
print(best_theta)

