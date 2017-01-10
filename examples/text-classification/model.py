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
from melt.models import Mlp

#this is input data related here is just demo usage,for our data has 34 classes and 324510 features
NUM_CLASSES = 34
NUM_FEATURES = 324510

def build_graph(X, y):
  #---build forward graph
  algo = Mlp(input_dim=NUM_FEATURES, num_classes=NUM_CLASSES)
  py_x = algo.forward(X)
  
  #-----------for classification we can set loss function and evaluation metrics,so only forward graph change
  #---set loss function
  loss = melt.sparse_softmax_cross_entropy(py_x, y)
  #tf.scalar_summary('loss_%s'%loss.name, loss)
 
  #---choose evaluation metrics
  accuracy = melt.precision_at_k(py_x, y, 1)
  #tf.scalar_summary('precision@1_%s'%accuracy.name, accuracy)

  return loss, accuracy
