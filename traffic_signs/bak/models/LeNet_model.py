from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import tensorflow as tf
import matplitlib.pyplot as plt
from sklearn.utils import shuffle


def get_weight_bias(shape, regularizer=None):
  weights = tf.get_variable(
    "weights", shape,
    initializer=tf.truncated_normal_initializer(stddev=0.1)
  )
  biases = tf.get_variable(
    "biases", (shape[-1]),
    initializer=tf.zeros((shape[-1]))
  )
  if regularizer:
    tf.add_to_collection("losses", regularizer(weights))

def valid_conv(inputs, weights, biases):
  inputs = tf.nn.conv2d(
    inputs, weights, strides=[1, 1, 1, 1],
    padding="VALID"
  )
  inputs = tf.bias_add(inputs, biases)
  return tf.nn.relu(inputs)

def inference(inputs, keep_prob):
  with tf.variable_scope("conv1-1"):
    weights, biases = get_weight_bias([3, 3, 1, 32])
    inputs = valid_conv      

  with tf.variable_scope("conv1-2"):
    weights, biases = get_weight_bias([3, 3, 32, 32])
    inputs = valid_conv      

  with tf.variable_scope("pool1"):
    inputs = tf.nn.max_pool(
      inputs, ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding="SAME"
    )    

  with tf.variable_scope("conv2-1"):
    weights, biases = get_weight_bias([3, 3, 32, 64])
    inputs = valid_conv      

  with tf.variable_scope("conv2-2"):
    weights, biases = get_weight_bias([3, 3, 64, 64])
    inputs = valid_conv      

  with tf.variable_scope("pool2"):
    inputs = tf.nn.max_pool(
      inputs, ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding="SAME"
    )    

  with tf.variable_scope("fc1"):
    weights, biases = get_weight_bias(
      [1600, 800], regularizer=regularizer
    )
    inputs = tf.nn.relu_layer(inputs, weights, biases)

  with tf.name_scope("dropout1"):
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

  with tf.variable_scope("fc2"):
    weights, biases = get_weight_bias(
      [800, 256], regularizer=regularizer
    )
    inputs = tf.nn.relu_layer(inputs, weights, biases)

  with tf.name_scope("dropout2"):
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

  with tf.variable_scope("logits"):
    weights, biases = get_weight_bias(
      [256, 43], regularizer=regularizer
    )
    inputs = tf.nn.relu_layer(inputs, weights, biases)

  return inputs