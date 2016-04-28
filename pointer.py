# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A pointer-network helper.
Based on attenton_decoder implementation from TensorFlow
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs


def pointer_decoder(decoder_inputs, initial_state, attention_states, cell,
                    feed_prev=True, dtype=dtypes.float32, scope=None):
    """RNN decoder with pointer net for the sequence-to-sequence model.
    Args:
      decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      dtype: The dtype to use for the RNN initial state (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "pointer_decoder".
    Returns:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of shape
        [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either i-th decoder_inputs.
        First, we run the cell
        on a combination of the input and previous attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      states: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        Each item is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())

    with vs.variable_scope(scope or "point_decoder"):
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        input_size = decoder_inputs[0].get_shape()[1].value
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])

        attention_vec_size = attn_size  # Size of query vectors for attention.
        k = vs.get_variable("AttnW", [1, 1, attn_size, attention_vec_size])
        hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
        v = vs.get_variable("AttnV", [attention_vec_size])

        states = [initial_state]

        def attention(query):
            """Point on hidden using hidden_features and query."""
            with vs.variable_scope("Attention"):
                y = rnn_cell.linear(query, attention_vec_size, True)
                y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                # Attention mask is a softmax of v^T * tanh(...).
                s = math_ops.reduce_sum(
                    v * math_ops.tanh(hidden_features + y), [2, 3])
                return s

        outputs = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = array_ops.zeros(batch_attn_size, dtype=dtype)

        attns.set_shape([None, attn_size])
        inps = []
        for i in xrange(len(decoder_inputs)):
            if i > 0:
                vs.get_variable_scope().reuse_variables()
            inp = decoder_inputs[i]

            if feed_prev and i > 0:
                inp = tf.pack(decoder_inputs)
                inp = tf.transpose(inp, perm=[1, 0, 2])
                inp = tf.reshape(inp, [-1, attn_length, input_size])
                inp = tf.reduce_sum(inp * tf.reshape(tf.nn.softmax(output), [-1, attn_length, 1]), 1)
                inp = tf.stop_gradient(inp)
                inps.append(inp)

            # Use the same inputs in inference, order internaly

            # Merge input and previous attentions into one vector of the right size.
            x = rnn_cell.linear([inp, attns], cell.input_size, True)
            # Run the RNN.
            cell_output, new_state = cell(x, states[-1])
            states.append(new_state)
            # Run the attention mechanism.
            output = attention(new_state)

            outputs.append(output)

    return outputs, states, inps
