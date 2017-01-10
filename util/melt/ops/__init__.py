import tensorflow as tf

#if int(tf.__version__.split('.')[1]) > 10:
#  from melt.ops.ops import *
#else:
#  from melt.ops.ops_backward_compat import *

#now only modify this , will have warning if tf 0.11
from melt.ops.ops_backward_compat import *

from melt.ops.sparse_ops import *

import melt.ops.seq2seq 
