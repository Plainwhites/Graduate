# LSTM-autoencoder

from reader import *
from LSTMAutoencoder import *


# Constants
# batch_num = 10
# hidden_num = 512
# step_num = 43
# elem_num = 60
iteration = 10000

# train data
folder = '/home/zy/Data/Florence/'
data = read_all_file(folder=folder)

config = Config(
    hidden_size=1024,
    elem_size=3*SKELETON_JOINT_NUMBER,
    step_size=43,
    layer_size=2,
    batch_size=10,
    keep_prob=0.5)

# placeholder list
p_input = tf.placeholder(
    tf.float32,
    shape=(
        config.batch_num,
        config.step_num,
        config.elem_num))
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, config.step_num, 1)]


ae = LSTMAutoencoder(config=config, inputs=p_inputs, is_training=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(iteration):
        """
        randomly generate the serial number of train data, index = [low, high)
        """
        t_input = random_input(data)

        (loss_val, _) = sess.run(
            [ae.loss, ae.train], {p_input: t_input})
        print('iter %d:' % (i + 1), loss_val)
