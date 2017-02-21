#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
from utils import *


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
H = 50
BATCH_SIZE = 100
DROP_OUT_RATE = 0.5


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Input: x : 28*28=784
x = tf.placeholder(tf.float32, [None, 784])

# Variable: W, b1
W1 = weight_variable((784, 256))
b1 = bias_variable([256])

fc1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = weight_variable((256, 50))
b2 = bias_variable([50])

# Hidden Layer: h
# softsign(x) = x / (abs(x)+1); https://www.google.co.jp/search?q=x+%2F+(abs(x)%2B1)
h = tf.nn.sigmoid(tf.matmul(fc1, W2) + b2)
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h, keep_prob)

# Variable: b2
W3 = tf.transpose(W2)  # 転置
b3 = bias_variable([256])
fc3 = tf.nn.relu(tf.matmul(h_drop, W3) + b3)

W4 = tf.transpose(W1)
b4 = bias_variable([784])

y = tf.nn.relu(tf.matmul(fc3, W4) + b4)

# Define Loss Function
loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

# For tensorboard learning monitoring
tf.summary.scalar("l2_loss", loss)

# Use Adam Optimizer
train_step = tf.train.AdamOptimizer().minimize(loss)

# Prepare Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
summary_writer = tf.summary.FileWriter('summary/l2_loss', graph_def=sess.graph_def)

# Training
for step in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    _, loss_str = sess.run([train_step, loss], feed_dict={x: batch_xs, keep_prob: (1-DROP_OUT_RATE)})
    # Collect Summary
    summary_op = tf.summary.merge_all()
    summary_str = sess.run(summary_op, feed_dict={x: batch_xs, keep_prob: 1.0})
    summary_writer.add_summary(summary_str, step)
    # Print Progress
    if step % 100 == 0:
        print(loss_str)

test_batch_xs, batch_ys = mnist.test.next_batch(100)
reconstruction = sess.run(y, feed_dict={x: test_batch_xs, keep_prob: 1.0})
visualize(test_batch_xs, reconstruction, 10)