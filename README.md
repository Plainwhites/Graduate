# Graduate
Work on my Master thesis



LSTMAutoencoder:
  reader.py is the file read the skeleton data which data format is txt.
  LSTMAutoencoder.py is the class of the model, uses the static lstm with fixed time steps.
  train_encoder.py: train the autoencoder.
DCNN+LSTM:
  read_video.py returns a list of videos, which shape is [batch_size, video_length, 640, 320, 3].
  inception_v1.py and inception_utils.py are copied from https://github.com/tensorflow/models/tree/master/research/slim/nets .
  dcnn_lstm.py is not accomplished.
  train_lstm_dcnn.py: train the multi-model.
