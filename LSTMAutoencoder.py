import tensorflow as tf


def data_type():
    """docstring"""

    return tf.float32


class LSTMAutoencoder(object):

    """
    """

    def __init__(self, config, inputs, is_training):
        """
        Args:

        """
        self.batch_num = config.batch_num
        self.elem_num = config.elem_num
        self.step_num = config.step_num
        self.hidden_num = config.hidden_num
        self.layer_num = config.layer_num

        self._enc_cell = self._build_rnn_graph_lstm_encoder(config, is_training)
        self._dec_cell = self._build_rnn_graph_lstm_decoder(config, is_training)

        # self._en_outputs = [] 暂不需要
        self._initial_state = self._enc_cell.zero_state(config.batch_num, data_type())
        self._enc_state = self._initial_state

        with tf.variable_scope('encoder'):
            for time_step in range(self.step_num):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, self._enc_state) = self._enc_cell(inputs[time_step], self._enc_state)

        with tf.variable_scope('decoder'):
            dec_weight_ = tf.Variable(tf.truncated_normal([self.hidden_num,
                                                           self.elem_num], dtype=data_type()), name='dec_weight')
            dec_bias_ = tf.Variable(tf.constant(0.1,
                                                shape=[self.elem_num],
                                                dtype=data_type()), name='dec_bias')
            self._dec_state = self._enc_state
            self._dec_input_ = tf.zeros(shape=[self.batch_num, self.elem_num], dtype=data_type())
            self._dec_outputs = []
            for step in range(config.step_num):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                (self._dec_input_, self._dec_state) = self._dec_cell(self._dec_input_, self._dec_state)
                self._dec_input_ = tf.matmul(self._dec_input_, dec_weight_) + dec_bias_
                self._dec_outputs.append(self._dec_input_)

            self._dec_outputs = self._dec_outputs[::-1]
            self.output_ = tf.transpose(tf.stack(self._dec_outputs), [1, 0, 2])

        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.loss = tf.reduce_mean(tf.square(self.input_-self.output_))

        self.train = tf.train.AdamOptimizer().minimize(self.loss)

    def _get_lstm_cell(self, config, is_training):
        """ the lstm cell """
        return tf.contrib.rnn.BasicLSTMCell(config.hidden_num,
                                            forget_bias=0.0,
                                            state_is_tuple=True,
                                            reuse=not is_training)

    def _build_rnn_graph_lstm_encoder(self, config, is_training):
        """Build the encoder inference graph using canonical LSTM cells."""

        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell

        enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.layer_num)], state_is_tuple=True)

        return enc_cell

    def _build_rnn_graph_lstm_decoder(self, config, is_training):
        """ Build decoder"""
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell

        de_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.layer_num)], state_is_tuple=True)

        return de_cell


class Config(object):
    """ Parameters are used to initialize the model """

    def __init__(self,
                 hidden_size,
                 elem_size,
                 step_size,
                 layer_size,
                 batch_size,
                 keep_prob
                 ):
        self.hidden_num = hidden_size
        self.elem_num = elem_size
        self.step_num = step_size
        self.layer_num = layer_size
        self.batch_num = batch_size
        self.keep_prob = keep_prob
