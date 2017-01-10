#!/usr/bin/env python
# ==============================================================================
#          \file   ops.py
#        \author   chenghuige  
#          \date   2016-08-16 10:09:34.992292
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#as tf_utils in order to avoid confilict/hidden melt.utils since (from melt.ops import * will import..)
#@NOTICE! be careful or if import uitls without rename, you need to from melt.ops not import * and use melt.ops..
from tensorflow.contrib.layers.python.layers import utils as tf_utils

def matmul(X, w):
  """ General matmul  that will deal both for dense and sparse input
  hide the differnce of dense and sparse input for end users
  Args:
  X: a tensor, or a list with two sparse tensors (index, value)
  w: a tensor
  """
  if not isinstance(X, (list, tuple)):
    return tf.matmul(X,w)
  else: 
    #X[0] index, X[1] value
    return tf.nn.embedding_lookup_sparse(w, X[0], X[1], combiner='sum')

#@TODO try to use slim.fully_connected
def mlp_forward(input, hidden, hidden_bais, out, out_bias, activation=tf.nn.relu, name=None):
  #@TODO **args?
  with tf.name_scope(name, 'mlp_forward', [input, hidden, hidden_bais, out, out_bias, activation]):
    hidden_output = activation(matmul(input, hidden) + hidden_bais)
    return tf.matmul(hidden_output, out) + out_bias

def mlp_forward_nobias(input, hidden, out, activation=tf.nn.relu, name=None):
  with tf.name_scope(name, 'mlp_forward_nobias', [input, hidden, out, activation]):
    hidden_output = activation(matmul(input, hidden))
    return tf.matmul(hidden_output, out)

def element_wise_cosine_nonorm(a, b, keep_dims=True, name=None):
  with tf.name_scope(name, 'element_wise_cosine_nonorm', [a, b]):
    return tf.reduce_sum(tf.mul(a, b), 1, keep_dims=keep_dims)

#[batch_size, y], [batch_size, y] => [batch_size, 1]
def element_wise_cosine(a, b, a_normed=False, b_normed=False, nonorm=False, keep_dims=True, name=None):
  if nonorm:
    return element_wise_cosine_nonorm(a, b, keep_dims, name)
  with tf.name_scope([a,b], name, 'element_wise_cosine'):
    if a_normed:
      normalized_a = a 
    else:
      normalized_a = tf.nn.l2_normalize(a, 1)
    if b_normed:
      normalized_b = b 
    else:
      normalized_b = tf.nn.l2_normalize(b, 1)
    #return tf.matmul(normalized_a, normalized_b, transpose_b=True)
    return tf.reduce_sum(tf.mul(normalized_a, normalized_b), 1, keep_dims=keep_dims)

def cosine_nonorm(a, b, name=None):
  with tf.name_scope(name, 'cosine_nonorm', [a,b]):
    return tf.matmul(a, b, transpose_b=True)  
#[batch_size, y] [x, y] => [batch_size, x]
def cosine(a, b, a_normed=False, b_normed=False, nonorm=False, name=None):
  if nonorm:
    return cosine_nonorm(a, b)
  with tf.name_scope(name, 'cosine', [a,b]):
    if a_normed:
      normalized_a = a 
    else:
      normalized_a = tf.nn.l2_normalize(a, 1)
    if b_normed:
      normalized_b = b 
    else:
      normalized_b = tf.nn.l2_normalize(b, 1)
    return tf.matmul(normalized_a, normalized_b, transpose_b=True)

def reduce_mean(input_tensor,  reduction_indices=None, keep_dims=False):
  """
  reduce mean with mask
  """
  return tf.reduce_sum(input_tensor, reduction_indices=reduction_indices, keep_dims=keep_dims) / \
         tf.reduce_sum(tf.sign(input_tensor), reduction_indices=reduction_indices, keep_dims=keep_dims)

def masked_reduce_mean(input_tensor,  reduction_indices=None, keep_dims=False, mask=None):
  """
  reduce mean with mask
  [1,2,3,0] -> 2 not 1.5 as normal
  """
  if mask is None:
    mask = tf.sign(input_tensor)
  return tf.reduce_sum(input_tensor, reduction_indices=reduction_indices, keep_dims=keep_dims) / \
         tf.reduce_sum(mask, reduction_indices=reduction_indices, keep_dims=keep_dims)

def reduce_mean_with_mask(input_tensor, mask, reduction_indices=None, keep_dims=False):
  return  tf.reduce_sum(input_tensor, reduction_indices=reduction_indices, keep_dims=keep_dims) / \
          tf.reduce_sum(mask, reduction_indices=reduction_indices, keep_dims=keep_dims)

def embedding_lookup(emb, index, reduction_indices=None, combiner='mean', name=None):
  with tf.name_scope(name, 'emb_lookup_%s'%combiner, [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    if combiner == 'mean':
      return tf.reduce_mean(lookup_result, reduction_indices)
    elif combiner == 'sum':
      return tf.reduce_sum(lookup_result, reduction_indices)
    else:
      raise ValueError('Unsupported combiner: ', combiner)

def embedding_lookup_mean(emb, index, reduction_indices=None, name=None):
  with tf.name_scope(name, 'emb_lookup_mean', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    return tf.reduce_mean(lookup_result, reduction_indices)

def embedding_lookup_sum(emb, index, reduction_indices=None, name=None):
  with tf.name_scope(name, 'emb_lookup_sum', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    return tf.reduce_sum(lookup_result, reduction_indices)

def masked_embedding_lookup(emb, index, reduction_indices=None, combiner='mean', exclude_zero_index=True, name=None):
  if combiner == 'mean':
    return masked_embedding_lookup_mean(emb, index, reduction_indices, combiner, exclude_zero_index, name)
  elif combiner == 'sum':
    return masked_embedding_lookup_sum(emb, index, reduction_indices, combiner, exclude_zero_index, name)
  else:
    raise ValueError('Unsupported combiner: ', combiner)

def masked_embedding_lookup_mean(emb, index, reduction_indices=None, exclude_zero_index=True, name=None):
  """
  @TODO need c++ op to really mask last dim zero feature vector
  now assume vector should zero filtered to be zero vector @TODO
  """
  with tf.name_scope(name, 'masked_emb_lookup_mean', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    if exclude_zero_index:
      masked_emb = mask2d(emb)
      mask_lookup_result = tf.nn.embedding_lookup(masked_emb, index)
      lookup_result = tf.mul(lookup_result, mask_lookup_result)
    return reduce_mean_with_mask(lookup_result,  
                                 tf.expand_dims(tf.cast(tf.sign(index), dtype=tf.float32), -1),
                                 reduction_indices)

def masked_embedding_lookup_sum(emb, index, reduction_indices=None, exclude_zero_index=True, name=None):
  """ 
  @TODO need c++ op to really mask last dim zero feature vector
  now assume vector should zero filtered  to be zero vector
  or to just make emb firt row zero before lookup ?
  """
  with tf.name_scope(name, 'masked_emb_lookup_sum', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    if exclude_zero_index:
      masked_emb = mask2d(emb)
      mask_lookup_result = tf.nn.embedding_lookup(masked_emb, index)
      lookup_result = tf.mul(lookup_result, mask_lookup_result)
    return tf.reduce_sum(lookup_result, reduction_indices)

def wrapped_embedding_lookup(emb, index, reduction_indices=None, combiner='mean', use_mask=False, name=None):
  """
  compare to embedding_lookup
  wrapped_embedding_lookup add use_mask
  """
  if use_mask:
    return masked_embedding_lookup(emb, index, reduction_indices, combiner, name)
  else:
    return embedding_lookup(emb, index, reduction_indices, combiner, name)

def batch_embedding_lookup(emb, index, combiner='mean', name=None):
  """
  same as embedding_lookup but use index_dim_length - 1 as reduction_indices
  """
  with tf.name_scope(name, 'batch_emb_lookup_%s'%combiner, [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    #@NOTICE for tf.nn.embedding_lookup, index can be list.. here only tensor
    reduction_indices = len(index.get_shape()) - 1
    if combiner == 'mean':
      return tf.reduce_mean(lookup_result, reduction_indices)
    elif combiner == 'sum':
      return tf.reduce_sum(lookup_result, reduction_indices)
    else:
      raise ValueError('Unsupported combiner: ', combiner)

def batch_embedding_lookup_mean(emb, index, name=None):
  with tf.name_scope(name, 'batch_emb_lookup_mean', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    reduction_indices = len(index.get_shape()) - 1
    return tf.reduce_mean(lookup_result, reduction_indices)

def batch_embedding_lookup_sum(emb, index, name=None):
  with tf.name_scope(name, 'batch_emb_lookup_sum', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    reduction_indices = len(index.get_shape()) - 1
    return tf.reduce_sum(lookup_result, reduction_indices)

def batch_masked_embedding_lookup(emb, index, combiner='mean', exclude_zero_index=True, name=None):
  if combiner == 'mean':
    return batch_masked_embedding_lookup_mean(emb, index, exclude_zero_index, name)
  elif combiner == 'sum':
    return batch_masked_embedding_lookup_sum(emb, index, exclude_zero_index, name)
  else:
    raise ValueError('Unsupported combiner: ', combiner)

def batch_masked_embedding_lookup_mean(emb, index, exclude_zero_index=True, name=None):
  """
  @TODO need c++ op to really mask last dim zero feature vector
  now assume vector should zero filtered to be zero vector if not exclude_zero_index
  or will have to do lookup twice
  """
  with tf.name_scope(name, 'batch_masked_emb_lookup_mean', [emb, index]):
    #if exclude_zero_index:
    #-----so slow..
    #  emb = tf.concat(0, [tf.zeros([1, emb.get_shape()[1]]), 
    #                      emb[1:, :]])
    lookup_result = tf.nn.embedding_lookup(emb, index)
    reduction_indices = len(index.get_shape()) - 1
    if exclude_zero_index:
      #@TODO this will casue 4 times slower 
      masked_emb = mask2d(emb)    
      mask_lookup_result = tf.nn.embedding_lookup(masked_emb, index)
      lookup_result = tf.mul(lookup_result, mask_lookup_result)
    return reduce_mean_with_mask(lookup_result,  
                                 tf.expand_dims(tf.cast(tf.sign(index), dtype=tf.float32), -1),
                                 reduction_indices)

def batch_masked_embedding_lookup_sum(emb, index, exclude_zero_index=True, name=None):
  """ 
  @TODO need c++ op to really mask last dim zero feature vector
  now assume vector should zero filtered to be zero vector if not exclude_zero_index
  or will have to do lookup twice
  """
  with tf.name_scope(name, 'batch_masked_emb_lookup_sum', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    reduction_indices = len(index.get_shape()) - 1  
    if exclude_zero_index:
      masked_emb = mask2d(emb)
      mask_lookup_result = tf.nn.embedding_lookup(masked_emb, index)
      lookup_result = tf.mul(lookup_result, mask_lookup_result)
    return tf.reduce_sum(lookup_result, reduction_indices)

def batch_wrapped_embedding_lookup(emb, index, combiner='mean', use_mask=False, exclude_zero_index=True, name=None):
  if use_mask:
    return batch_masked_embedding_lookup(emb, index, combiner, exclude_zero_index, name)
  else:
    return batch_embedding_lookup(emb, index, combiner, name)

def mask2d(emb):
  return tf.concat(0, [tf.zeros([1, 1]), 
                       tf.ones([emb.get_shape()[0] - 1, 1])])   

def length(x, dim=1):
  return tf.reduce_sum(tf.sign(x), dim)

#---------now only consider 2d @TODO
def dynamic_append(x, value=1):
  length = tf.reduce_sum(tf.sign(x), 1)
  rows = tf.range(tf.shape(x)[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.pack([rows, length]))
  shape = tf.cast(tf.shape(x), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def dynamic_append_with_mask(x, mask, value=1):
  length = tf.reduce_sum(mask, 1)
  rows = tf.range(tf.shape(x)[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.pack([rows, length]))
  shape = tf.cast(tf.shape(x), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def dynamic_append_with_length(x, length, value=1):
  rows = tf.range(tf.shape(x)[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.pack([rows, length]))
  shape = tf.cast(tf.shape(x), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

#@TODO not tested
def static_append(x, value=1):
  length = tf.reduce_sum(tf.sign(x), 1)
  rows = tf.range(x.get_shape()[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.pack([rows, length]))
  shape = tf.cast(x.get_shape(), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def static_append_with_mask(x, mask, value=1):
  length = tf.reduce_sum(mask, 1)
  rows = tf.range(x.get_shape()[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.pack([rows, length]))
  shape = tf.cast(x.get_shape(), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def static_append_with_length(x, length, value=1):
  rows = tf.range(x.get_shape()[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.pack([rows, length]))
  shape = tf.cast(x.get_shape(), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta


def first_nrows(x, n):
  #eval_scores[0:num_evaluate_examples, 1] the diff is below you do not need other dim infos
  return tf.gather(x, range(n))

def exclude_last_col(x):
  """
  just hack since dynamic x[:,:-1] is not supported 
  now just work for 2d
  ref to https://github.com/tensorflow/tensorflow/issues/206
  now tf.11 support X[:,-1]  so most likely [:,:-1] will also be ok ? TODO  check and change, but may be still need to compact for tf10 for hadoop predict
  """
  return tf.transpose(tf.gather(tf.transpose(x), tf.range(0, x.get_shape()[1] - 1)))

def dynamic_exclude_last_col(x):
  """
  @TODO just hack since dynamic x[:,:-1] is not supported 
  now just work for 2d
  ref to https://github.com/tensorflow/tensorflow/issues/206
  """
  return tf.transpose(tf.gather(tf.transpose(x), tf.range(0, tf.shape(x)[1] - 1)))

def gather2d(x, idx):
  """
  from https://github.com/tensorflow/tensorflow/issues/206
  x = tf.constant([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
idx = tf.constant([1, 0, 2])
idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
y = tf.gather(tf.reshape(x, [-1]),  # flatten input
              idx_flattened)  # use flattened indices

with tf.Session(''):
  print y.eval()  # [2 4 9]
  """
  #FIXME
  idx_flattened = tf.cast(tf.range(0, x.shape[0]) * x.shape[1], idx.dtype) + idx
  y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                idx_flattened)  # use flattened indices
  return y

def dynamic_gather2d(x, idx):
  #FIMXE
  idx_flattened = tf.cast(tf.range(0, tf.shape(x)[0]) * tf.shape(x)[1], idx.dtype) + idx
  y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                idx_flattened)  # 
  return y

def subtract_by_diff(x, y):
  """
  [1,2, 3, 4] - [1, 2, 1, 4] = [1, 2, 2, 4]
  assume input x, y is not the same
  @TODO c++ op
  """
  delta = tf.abs(x - y)
  delta_bool = tf.cast(delta, tf.bool)
  return tf.add(tf.mul(x, tf.cast(tf.logical_not(delta_bool), x.dtype)), delta)

#------this can only deal with same first dimension.. if two batch different batch size not ok
# def _align(x, y, dim):
#   x_shape = tf.shape(x)
#   y_shape = tf.shape(y)
#   padding_shape = subtract_by_diff(x_shape, y_shape)
#   x, y = tf.cond(
#     tf.greater(x_shape[dim], y_shape[dim]), 
#     lambda: (x, tf.concat(dim, [y, tf.zeros(padding_shape, x.dtype)])), 
#     lambda: (tf.concat(dim, [x, tf.zeros(padding_shape, x.dtype)]), y))
#   return x, y

# def align(x, y, dim):
#   """
#   @TODO use c++ op
#   """
#   x_shape = tf.shape(x)
#   y_shape = tf.shape(y)
#   x, y = tf.cond(
#     tf.equal(x_shape[dim], y_shape[dim]),
#     lambda: (x, y),
#     lambda: _align(x, y, dim))
#   return x, y


def _align_col_padding2d(x, y):
  x_shape = tf.shape(x)
  y_shape = tf.shape(y)

  x, y = tf.cond(
    tf.greater(x_shape[1], y_shape[1]), 
    lambda: (x, tf.pad(y, [[0, 0], [0, x_shape[1] - y_shape[1]]])), 
    lambda: (tf.pad(x, [[0, 0], [0, y_shape[1] - x_shape[1]]]), y))
  return x, y

def align_col_padding2d(x, y):
  x_shape = tf.shape(x)
  y_shape = tf.shape(y)
  x, y = tf.cond(
    tf.equal(x_shape[1], y_shape[1]),
    lambda: (x, y),
    lambda: _align_col_padding2d(x, y))
  return x, y

def last_relevant(output, length):
  """
  https://danijar.com/variable-sequence-lengths-in-tensorflow/
  Select the Last Relevant Output
  For sequence classification, we want to feed the last output of the recurrent network into a predictor, e.g. a softmax layer. While taking the last frame worked well for fixed-sized sequences, we not have to select the last relevant frame. This is a bit cumbersome in TensorFlow since it does’t support advanced slicing yet. In Numpy this would just be output[:, length - 1]. But we need the indexing to be part of the compute graph in order to train the whole system end-to-end.
  @TODO understand below code
  """
  batch_size = tf.shape(output)[0]
  #@TODO could not use in rnn.py why even int fixed length mode?  max_length = int(output.get_shape()[1]) __int__ returned non-int (type NoneType) 
  #because even though you convert sparse to dense to make same length, that length is dynamic , if get in static will be None? So if you use FixedLen to read tfrecord like version 0 in 
  #models/image-text-sim then might ok here, but for comment seems using feed_dict mode still None why? placeholder should have fixed length! 
  #@TODO why need int otherwise in tf.reshape(output, [-1, out_size]) TypeError: Expected int32, got Dimension(1024) of type 'Dimension' instead.
  max_length = int(output.get_shape()[1])
  out_size = int(output.get_shape()[2])
  #index = tf.range(0, batch_size) * max_length + (length - 1)
  #@TODO may be it is best to convert tfrecord reading to int64 and convert to int32 to avoid unnecessary cast
  index = tf.cast(tf.range(0, batch_size), length.dtype) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant


#----------for rnn
def dynamic_last_relevant(output, length):
  """
  https://danijar.com/variable-sequence-lengths-in-tensorflow/
  Select the Last Relevant Output
  For sequence classification, we want to feed the last output of the recurrent network into a predictor, e.g. a softmax layer. While taking the last frame worked well for fixed-sized sequences, we not have to select the last relevant frame. This is a bit cumbersome in TensorFlow since it does’t support advanced slicing yet. In Numpy this would just be output[:, length - 1]. But we need the indexing to be part of the compute graph in order to train the whole system end-to-end. 

  not this will only work for 3 d, for general pupose dynamic last might consider
  output = tf.reverse_sequence(output, seqence_lenth, 1)
  return output[:, 0, :]
  """
  shape = tf.shape(output)
  batch_size = shape[0]
  #max_length = shape[1]
  max_length = tf.cast(shape[1], length.dtype)
  #out_size = shape[2]
  out_size = int(output.get_shape()[2])
  #index = tf.range(0, batch_size) * max_length + (length - 1)
  #@TODO may be it is best to convert tfrecord reading to int64 and convert to int32 to avoid unnecessary cast
  #here length might be tf.int64 if length calced from mask of int64 type
  index = tf.cast(tf.range(0, batch_size), length.dtype) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant

def dynamic_last(output):
  """
  https://danijar.com/variable-sequence-lengths-in-tensorflow/
  Select the Last Relevant Output
  For sequence classification, we want to feed the last output of the recurrent network into a predictor, e.g. a softmax layer. While taking the last frame worked well for fixed-sized sequences, we not have to select the last relevant frame. This is a bit cumbersome in TensorFlow since it does’t support advanced slicing yet. In Numpy this would just be output[:, length - 1]. But we need the indexing to be part of the compute graph in order to train the whole system end-to-end.
  """
  shape = tf.shape(output)
  batch_size = shape[0]
  max_length = shape[1]
  #out_size = shape[2]
  out_size = int(output.get_shape()[2])
  #index = tf.range(0, batch_size) * max_length + (length - 1)
  #@TODO may be it is best to convert tfrecord reading to int64 and convert to int32 to avoid unnecessary cast
  index = tf.range(0, batch_size) * max_length + (max_length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant

def static_last(output):
  return output[:, int(output.get_shape()[1]) - 1, :]

#-----------loss ops
#@TODO contrib\losses\python\losses\loss_ops.py
def sparse_softmax_cross_entropy(x, y):
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(x, y))

def softmax_cross_entropy(x, y):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(x, y))

def sigmoid_cross_entropy(x, y):
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x, y))

activations = {'sigmoid' :  tf.nn.sigmoid, 'tanh' : tf.nn.tanh, 'relu' : tf.nn.relu}

#--@TODO other rank loss
#@TODO move to losses
def hinge_loss(pos_score, neg_score, margin=0.1, combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_loss', [pos_score, neg_score]):
    loss_matrix = tf.maximum(0., margin - (pos_score - neg_score))
    if combiner == 'mean':
      loss = tf.reduce_mean(loss_matrix)
    else:
      loss = tf.reduce_sum(loss_matrix)
    return loss

def last_dimension(x):
  return tf_utils.last_dimension(x.get_shape())

def first_dimension(x):
  return tf_utils.first_dimension(x.get_shape())

def dimension(x, index):
  return x.get_shape()[index].value