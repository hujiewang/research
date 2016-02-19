import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn import _reverse_seq
import numpy as np

def rnn_return_states(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
    outputs = []
    states = []
    with vs.variable_scope(scope or "RNN"):
        batch_size = 1
        state = initial_state

        if sequence_length:  # Prepare variables
            zero_output = array_ops.zeros(
                array_ops.pack([batch_size, cell.output_size]), inputs[0].dtype)
            zero_state = array_ops.zeros(
                array_ops.pack([batch_size, cell.state_size]), inputs[0].dtype)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0: vs.get_variable_scope().reuse_variables()
            # pylint: disable=cell-var-from-loop
            output_state = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            if sequence_length:
                (output, state) = control_flow_ops.cond(
                    time >= max_sequence_length,
                    lambda: (zero_output, zero_state), output_state)
            else:
                (output, state) = output_state()

            outputs.append(output)
            states.append(state)


    return (outputs, states)

def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):
    name = scope or "BiRNN"
    # Forward direction
    with vs.variable_scope(name + "_FW",initializer=tf.constant_initializer(0.005)):
        _, state_fw = rnn_return_states(cell_fw, inputs, initial_state_fw, dtype,
                        sequence_length)

    # Backward direction
    with vs.variable_scope(name + "_BW",initializer=tf.constant_initializer(0.005)):
        _, tmp = rnn_return_states(cell_bw, _reverse_seq(inputs, sequence_length),
                    initial_state_bw, dtype, sequence_length)
        state_bw = _reverse_seq(tmp, sequence_length)
        # Concat each of the forward/backward outputs
        states = [array_ops.concat(1, [fw, bw]) for fw, bw in zip(state_fw, state_bw)]

    return states


class DefaultConfig:
    def __init__(self):
        self.max_length_0_input=3
        self.max_length_1_input=3
        self.embedding_size=3

class Model:
    def __init__(self, config=DefaultConfig()):
        self.config = config
        self.session = tf.Session()
    def build(self):

        self.input_0=tf.placeholder(tf.float32,[self.config.max_length_0_input,1,self.config.embedding_size])
        self.input_0_length=tf.placeholder(tf.int32)

        self.input_1=tf.placeholder(tf.float32,[self.config.max_length_0_input,1,self.config.embedding_size])
        self.input_1_length=tf.placeholder(tf.int32)

        input_0=array_ops.unpack(self.input_0)
        input_1=array_ops.unpack(self.input_1)

        # bidirectional rnn
        cell=rnn_cell.GRUCell(self.config.embedding_size)

        initial_state_fw=array_ops.zeros(array_ops.pack([1, cell.state_size]), dtype=tf.float32)
        initial_state_fw.set_shape([1, cell.state_size])
        initial_state_bw=array_ops.zeros(array_ops.pack([1, cell.state_size]), dtype=tf.float32)
        initial_state_bw.set_shape([1, cell.state_size])

        states=bidirectional_rnn(cell, cell, input_0,
                                 initial_state_fw=initial_state_fw,
                                 initial_state_bw=initial_state_bw,
                                 dtype=tf.float32,
                                 # sequence_length=3
                                 )


        self.test=array_ops.pack(states)

    def fit(self, X_0, X_1, y, meta):
        '''
        X_shape: N x length x emb_size
        '''


        init_op = tf.initialize_all_variables()
        self.session.run(init_op)

        states=self.session.run([self.test],{self.input_0: X_0,self.input_1: X_1})
        print(states[0])



X_0=np.array([[[1,2,3]],[[4,5,6]],[[1,2,3]]])
X_1=np.array([[[1,2,3]],[[4,5,6]],[[7,8,9]]])
y=np.array([0])
meta=[3,3]

m=Model()
m.build()
m.fit(X_0,X_1,y,meta)



