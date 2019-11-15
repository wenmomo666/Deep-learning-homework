#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


class MyGRUCell(RNNCell):
    """
    Your own basic GRUCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow GRUCell source code. If you're using Anaconda, it's located at
    anaconda_install_path/envs/your_virtual_environment_name/site-packages/tensorflow/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow GRUCell, but with your own language.
    """

    def __init__(self, num_units,activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the GRU cell.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyGRUCell, self).__init__(_reuse=True)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        self.num_units_=num_units
        self.activation_=activation or tf.tanh
        
        with tf.name_scope('GRU.Weight'):
            self.w_z_=tf.Variable(tf.random_normal([num_units+1,num_units]),name='w_z')
            self.b_z_=tf.Variable(tf.ones([num_units],name='b_z'))
            self.w_r_=tf.Variable(tf.random_normal([num_units+1,num_units]),name='w_r')
            self.b_r_=tf.Variable(tf.ones([num_units],name='b_r'))
            self.w_h_=tf.Variable(tf.random_normal([num_units+1,num_units]),name='w_h')
            self.b_h_=tf.Variable(tf.ones([num_units],name='b_h'))



    

    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units =output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        return self.num_units_

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        return self.num_units_

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow GRUCell has a lot of other features. But we will not try them. Focus on the
        very basic GRU functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In GRU there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function
                       state_size(self).
        :return: A tuple containing (new_state, new_state). For details check TensorFlow GRUCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        z=tf.sigmoid(tf.nn.bias_add(tf.matmul(tf.concat([state,inputs],axis=1),self.w_z_),self.b_z_))
        r=tf.sigmoid(tf.nn.bias_add(tf.matmul(tf.concat([state,inputs],axis=1),self.w_r_),self.b_r_))
        a=self.activation_(tf.nn.bias_add(tf.matmul(tf.concat([r*state,inputs],axis=1),self.w_h_),self.b_h_))
        state1=z*state+(1-z)*a
        return state1, state1
            

