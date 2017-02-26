#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2016-08-19 01:31:54.834381
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import melt
from melt.models import mlp


g_num_features = -1

def set_input_info(num_features):
  global g_num_features
  g_num_features = num_features

def predict(X):
  return mlp.forward(X, 
                    input_dim=g_num_features, 
                    num_outputs=1, 
                    hiddens=[200,100,50])
                    #hiddens=[200])

def build_graph(X, y):
  #---build forward graph
  py_x = predict(X)
  
  #-----------for classification we can set loss function and evaluation metrics,so only forward graph change
  #---set loss function
  
  #TODO: not work shapes (64, ?) (?, ?) not incompatibale
  #loss = tf.losses.mean_squared_error(py_x, y)
  loss = tf.reduce_mean(tf.pow(py_x - y, 2))

  #---choose evaluation metrics
  correct_prediction = tf.equal(tf.cast(py_x, tf.int32), tf.cast(y, tf.int32))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return loss, accuracy
