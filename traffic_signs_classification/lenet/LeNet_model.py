from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


def get_weight_bias(shape):
  weights = tf.get_variable(
    "weights", shape=shape,
    initializer=tf.truncated_normal_initializer(stddev=0.1)
  )
  biases = tf.get_variable(
    "biases", shape=[shape[-1]],
    initializer=tf.constant_initializer(.0)
  )
  return weights, biases

def valid_conv(inputs, weights, biases):
  inputs = tf.nn.conv2d(
    inputs, weights, strides=[1, 1, 1, 1],
    padding="VALID"
  )
  inputs = tf.nn.bias_add(inputs, biases)
  return tf.nn.relu(inputs)

def inference(inputs, keep_prob, regularizer):

  with tf.variable_scope("conv1-1"):
    weights, biases = get_weight_bias([3, 3, 1, 32])
    inputs = valid_conv(inputs, weights, biases)     

  with tf.variable_scope("conv1-2"):
    weights, biases = get_weight_bias([3, 3, 32, 32])
    inputs = valid_conv(inputs, weights, biases)     

  with tf.name_scope("pool1"):
    inputs = tf.nn.max_pool(
      inputs, ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding="VALID"
    )    

  with tf.variable_scope("conv2-1"):
    weights, biases = get_weight_bias([3, 3, 32, 64])
    inputs = valid_conv(inputs, weights, biases)     

  with tf.variable_scope("conv2-2"):
    weights, biases = get_weight_bias([3, 3, 64, 64])
    inputs = valid_conv(inputs, weights, biases)     

  with tf.name_scope("pool2"):
    inputs = tf.nn.max_pool(
      inputs, ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding="VALID"
    )    

  inputs = tf.contrib.layers.flatten(inputs)

  with tf.variable_scope("fc1"):
    weights, biases = get_weight_bias(
      [1600, 800]
    )
    inputs = tf.nn.relu_layer(inputs, weights, biases)

  with tf.name_scope("dropout1"):
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

  with tf.variable_scope("fc2"):
    weights, biases = get_weight_bias(
      [800, 256]
    )
    inputs = tf.nn.relu_layer(inputs, weights, biases)

  with tf.name_scope("dropout2"):
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

  with tf.variable_scope("logits"):
    weights, biases = get_weight_bias(
      [256, 43]
    )
    inputs = tf.nn.relu_layer(inputs, weights, biases)

  return inputs
