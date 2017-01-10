#!/usr/bin/env python
# ==============================================================================
#          \file   mlp.py
#        \author   chenghuige  
#          \date   2016-08-16 17:13:04.699501
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

import melt

#@TODO muliti layer mlp using slim.stack
#@TODO will use slim.fully_connected and other layers to replace but right now slim.fully_connected seems not show same result, overfit to 1..
#@TODO no need for input dim 
class Mlp(object):
 def __init__(self, input_dim, num_classes, hidden_size=200, activation=tf.nn.relu):
   self.input_dim = input_dim
   self.num_classes = num_classes 
   self.hidden_size = hidden_size
   self.activation = activation

 def forward(self, X):
   hidden_size = self.hidden_size
   input_dim = self.input_dim 
   num_classes = self.num_classes

   #--@TODO slim.stack
   # Verbose way:
   #x = slim.fully_connected(x, 32, scope='fc/fc_1')
   #x = slim.fully_connected(x, 64, scope='fc/fc_2')
   #x = slim.fully_connected(x, 128, scope='fc/fc_3')
   ## Equivalent, TF-Slim way using slim.stack:
   #slim.stack(x, slim.fully_connected, [32, 64, 128], scope='fc')
   
   #--------so no local variable, make it get_var or class member var so we can share
   if isinstance(X, tf.Tensor):
     w_h = melt.get_weights('w_h', [input_dim, hidden_size])
   else:
     with tf.device('/cpu:0'):
       w_h = melt.get_weights('w_h', [input_dim, hidden_size]) 
   b_h = melt.get_bias('b_h', [hidden_size])
   w_o = melt.get_weights('w_o', [hidden_size, num_classes])
   b_o = melt.get_bias('b_o', [num_classes])
   py_x = melt.mlp_forward(X, w_h, b_h, w_o, b_o, activation=self.activation)

   return py_x  
