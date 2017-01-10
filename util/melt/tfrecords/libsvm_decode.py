#!/usr/bin/env python
# ==============================================================================
#          \file   libsvm_decode.py
#        \author   chenghuige  
#          \date   2016-08-15 20:17:53.507796
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#notice heare use parse_example not parse single example for it
#is used in sparse record reading flow, suffle then deocde
def decode(batch_serialized_examples):
  """
  decode batch_serialized_examples for use in parse libsvm fomrat sparse tf-record
  Returns:
  X,y
  """
  features = tf.parse_example(
      batch_serialized_examples,
      features={
          'label' : tf.FixedLenFeature([], tf.int64),
          'index' : tf.VarLenFeature(tf.int64),
          'value' : tf.VarLenFeature(tf.float32),
      })

  label = features['label']
  index = features['index']
  value = features['value']

  #return as X,y
  return (index, value), label
