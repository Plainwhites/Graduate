from read_video import *
from dcnn_lstm import *

rnn_batch_size = 10
class_number = 9

dcnn_input = tf.placeholder(dtype=tf.float32, shape=[rnn_batch_size, None, 640, 380, 3])
labels = tf.placeholder(dtype=tf.float32, shape=[rnn_batch_size, class_number])
# num_frame = tf.placeholder(dtype=tf.float32, shape=[rnn_batch_size, 1])
num_frame = list(np.random.randint(low=10, high=30, size=10))
dcnnlstm = Dcnnlstm(inputs=dcnn_input, labels=labels, num_frame=num_frame)

iteration = 10000

video_folder = '/home/zy/Data/Florence_3d_actions/'

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "/home/zy/Data/inception_v1.ckpt")

    for i in range(iteration):
        random_input_video, random_label, num_frames = get_random_video_frames(video_folder)
        (loss_val, _) = sess.run([dcnnlstm.loss, dcnnlstm.train],
                                 {dcnn_input: random_input_video,
                                  labels: random_label,
                                  num_frame: num_frames})
        print('iter %d:' % (i + 1), loss_val)
