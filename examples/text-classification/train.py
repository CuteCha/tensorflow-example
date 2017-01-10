#!/usr/bin/env python
# ==============================================================================
#          \file   train-melt.py
#        \author   chenghuige  
#          \date   2016-08-16 14:05:38.743467
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')

import melt
import model
decode = melt.libsvm_decode.decode

def train():
  trainset = sys.argv[1]
  inputs = melt.shuffle_then_decode.inputs
  X, y = inputs(
    trainset, 
    decode=decode,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs, 
    num_threads=FLAGS.num_preprocess_threads,
    batch_join=FLAGS.batch_join,
    shuffle=FLAGS.shuffle)
  
  train_with_validation = len(sys.argv) > 2
  if train_with_validation:
    validset = sys.argv[2]
    eval_X, eval_y = inputs(
      validset, 
      decode=decode,
      batch_size=FLAGS.batch_size * 10,
      num_threads=FLAGS.num_preprocess_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.shuffle)
  
  loss, accuracy = model.build_graph(X, y)
  if train_with_validation:
    tf.get_variable_scope().reuse_variables()
    eval_loss, eval_accuracy = model.build_graph(eval_X, eval_y)
    eval_ops = [eval_loss, eval_accuracy]
  else:
    eval_ops = None

  melt.apps.train_flow(
             [loss, accuracy], 
             deal_results=melt.show_precision_at_k,
             eval_ops=eval_ops,
             deal_eval_results= lambda x: melt.print_results(x, names=['precision@1']),
             )

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
