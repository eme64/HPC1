#!/usr/local/bin/python
'''
  *
  *  written by Guido Novati: novatig@ethz.ch
  *  Copyright 2017 ETH Zurich. All rights reserved.
  *
'''
import tensorflow as tf
import numpy as np
import sys, time
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape, name, bRestart=False):
  if bRestart:
    initial = np.fromfile(name+".raw", dtype=np.float64).reshape(shape)
  else: initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape, name, bRestart=False):
  if bRestart:
    initial = np.fromfile(name+".raw", dtype=np.float64).reshape(shape)
  else: initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

nepochs = 50
batchsize = 32
x = tf.placeholder(tf.float32, shape=[None, 28*28])
#y_ = tf.placeholder(tf.float32, shape=[None, 10])


W_fc1 = weight_variable([28*28, 10], "W_fc1")
b_fc1 = bias_variable([10], "b_fc1")
h_fc1 = tf.matmul(x, W_fc1) + b_fc1

W_fc2 = weight_variable([10, 28*28], "W_fc2")
b_fc2 = bias_variable([28*28], "b_fc2")
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

loss = tf.nn.l2_loss(x-h_fc2)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

steps_in_epoch = mnist.train.num_examples // batchsize
test_size = mnist.test.num_examples
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

#with tf.Session(config=session_conf) as sess: # single threaded
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(nepochs):
    for j in range(steps_in_epoch):
      batch = mnist.train.next_batch(batchsize)
      train_step.run(feed_dict={x: batch[0]})
    
    if i % 10 == 0:
      err = loss.eval(feed_dict={x:mnist.test.images})/test_size
      print('test error %g' % err)
  
  W_eval = W_fc2.eval()# retrieve from tf
  b_eval = b_fc2.eval()
  b_eval = np.reshape(b_eval, (784, 1))
  
  print("<><><><>")
  for i in range(0,10):
    print('saving %g' % i)
    ei = np.zeros((10,1)) ## unit vector
    ei[i, 0] = 1.0
    print(W_eval.shape)
    print(ei.shape)
    print(b_eval.shape)
    result = np.dot(W_eval.T, ei) + b_eval
    result = np.reshape(result, (28, 28)) # put image on screen
    print(result.shape)
    plt.imshow(result)
    plt.show()
    
