import tensorflow as tf
import inception_v1
import LSTMAutoencoder


rnn_batch_size = 10
class_number = 9


class Dcnnlstm(object):
    """class of network"""

    def __init__(self, inputs, labels, num_frame):
        # inputs is a placeholder  shape=[batch_size, None, 640, 380, 3]
        self.input = inputs
        self.feature_maps = []
        self.label = labels
        self.label = labels
        # load the parameters in the session
        # calculate the convs
        # self.net, self.endpoints = inception_v1.inception_v1_base(inputs=self.input,
        #                                                           final_endpoint='Mixed_5c',
        #                                                           scope='InceptionV1')

        # prepare the feature map of videos

        for i in range(self.input.shape[0]):
            tmp = []
            for j in range(num_frame[i]):
                net = inception_v1.inception_v1_base(inputs=self.input[i][j],
                                                     final_endpoint='Mixed_5c',
                                                     scope='InceptionV1')
                net = tf.reshape(net, shape=[1, -1])
                tmp.append(net)
            self.feature_maps.append(tmp)

        self._rnn_ = self.get_dynamic_lstm()
        self._initial_state_ = self._rnn_.zero_state(rnn_batch_size, LSTMAutoencoder.data_type())
        self.state = self._initial_state_
        self._output, self.state = tf.nn.dynamic_rnn(self._rnn_,
                                                     inputs=self.feature_maps,
                                                     dtype=LSTMAutoencoder.data_type())

        # self.output's shape is [batch_size, max_video_length, 1024]
        # then we must get each last cell's output of the videos

        self.lstm_output = []
        for i in range(self.input.shape[0]):
            j = self.input[i].shape[0]
            self.lstm_output.append(self._output[i][j-1])

        weight = tf.Variable(tf.truncated_normal([1024, 9], mean=0.1, stddev=1.0, dtype=tf.float32))
        bias = tf.Variable(tf.truncated_normal(shape=[10], mean=0.1, stddev=1.0, dtype=tf.float32))

        y_ = []
        for _ in self.input.shape[0]:
            y_.append(tf.matmul(self.lstm_output[_], weight) + bias)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, self.label))

        self.train = tf.train.GradientDescentOptimizer(0.9).minimize(self.loss)

    def get_dynamic_lstm(self):
        lstm_1 = tf.contrib.rnn.BasicLSTMCell(2048,
                                              forget_bias=0.0,
                                              state_is_tuple=True,
                                              reuse=True)
        lstm_1 = tf.contrib.rnn.DropoutWrapper(lstm_1, output_keep_prob=0.8)

        lstm_2 = tf.contrib.rnn.BasicLSTMCell(1024,
                                              forget_bias=0.0,
                                              state_is_tuple=True,
                                              reuse=True)
        lstm_2 = tf.contrib.rnn.DropoutWrapper(lstm_2, output_keep_prob=0.8)

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_1, lstm_2], state_is_tuple=True)

        return multi_rnn_cell
