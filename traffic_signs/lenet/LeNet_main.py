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

from traffic_signs.lenet import LeNet_model
from traffic_signs.lenet import data_loader



# Hyper parameters
_DATA_DIR = {
  "train": "/home/eric/Projects/self-driving/traffic-signs-data/train.p",
  "valid": "/home/eric/Projects/self-driving/traffic-signs-data/valid.p",
  "test":  "/home/eric/Projects/self-driving/traffic-signs-data/test.p"
}
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = .003
REGULARIZATION_RATE = 0.00005

X_train, y_train, X_valid, y_valid, X_test, y_test = data_loader.data_loader(_DATA_DIR)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

y_one_hot = tf.one_hot(y, 43)
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
logits = LeNet_model.inference(
  x, keep_prob=keep_prob, 
  regularizer=regularizer
  )

cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
  labels=y_one_hot, logits=logits
)
regularization_loss = tf.add_n(tf.get_collection("losses"))
loss_operation = tf.reduce_mean(cross_entropy_loss) + regularization_loss

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
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
        keep_prob: 1.
      }
    )
    total_acc += acc * len(batch_x)

  return total_acc / number_examples


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  number_examples = len(X_train)

  print("Start training ...")
  print()
  time0 = time()
  for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, number_examples, BATCH_SIZE):
      batch_x, batch_y = X_train[offset: offset + BATCH_SIZE], \
                         y_train[offset: offset + BATCH_SIZE]
      sess.run(
        training_operation, 
        feed_dict={
          x: batch_x,
          y: batch_y,
          keep_prob: .5
        })

    validation_acc = evaluate(X_valid, y_valid)
    print("Epoch {} ...".format(i+1))
    print("Validation accuracy = {:.3f}".format(validation_acc))
    print()
  time1 = time()
  saver.save(sess, "../saved_models")
  print("Model saved!")
  print("{}s taken to train the model.\n".format((time1 - time0) * 100))
    
