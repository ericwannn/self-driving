from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from time import time

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from traffic_sign_classification.lenet import data_loader
from traffic_sign_classification.lenet import LeNet_model

# Hyper parameters
_DATA_DIR = {
  "train": "../traffic_signs_data/train.p",
  "valid": "../traffic_signs_data/valid.p",
  "test":  "../traffic_signs_data/test.p"
}
EPOCHS = 20
BATCH_SZIE = 128
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
y_one_hot = tf.placeholder(y, 43)

X_train, y_train, X_valid, y_valid, X_test, y_test = data_loader.data_loader(_DATA_DIR)

rate = .001
logits = LeNet_model.inference(x)
cross_entroy = tf.nn.softmax_cross_entropy_with_logits(
  labels=y_one_hot, logits=logits
)

loss_operation = tf.reduce_mean(cross_entroy)

optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(
  tf.argmax(logits, 1), tf.argmax(y_one_hot, 1)
  )
accuracy_operation = tf.reduce_mean(
  tf.cast(correct_prediction, tf.float32)
)
saver = tf.train.Saver()

def evaluate(X_data, y_data):
  number_examples = len(X_data)
  total_acc = 0

  sess = tf.get_default_session()
  for offset in range(0, number_examples, BATCH_SIZE):
    batch_x, batch_y = X_data[offset: offset + BATCH_SIZE], \
                       y_data[offset: offset + BATCH_SIZE]
    acc = sess.run(
      accuracy_operation, 
      feed_dict={
        x: batch_x,
        y: batch_y,
        keep_prob: .5
      }
    )
    total_acc += acc * len(batch_x)

  return total_acc / number_examples


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  number_examples = len(X_train)

  print("Start training ...")
  print()

  for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, number_examples, BATCH_SIZE):
      batch_x, batch_y = X_train[offset: offset + BATCH_SZIE], \
                         y_train[offset: offset + BATCH_SIZE]
      sess.run(
        training_operation, 
        feed_dict={
          x: batch_x,
          y: batch_y,
          keep_prob: .5
        })

    validation_acc = evaluate(X_valid, y_valid, keep_prob=1.)
    print("Epoch {} ...".format(i+1))
    print("Validation accuracy = {:.3f}".format(validation_acc))
    print()

  saver.save(sess, "../saved_models")
  print("Model saved!")
    
