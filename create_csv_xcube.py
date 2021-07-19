import rnn_ppg
import numpy as np
import tensorflow as tf

rnn_ppg.subjs_test = (5, 6)

for decimation in (0, 10, 20, 30, 40, 50, 60, 80, 100, 150):
	rnn_ppg.window = 1200
	rnn_ppg.overlap = 1200 // 2
	rnn_ppg.decimation = decimation
	if rnn_ppg.decimation:
		rnn_ppg.window //= rnn_ppg.decimation
		rnn_ppg.overlap //= rnn_ppg.decimation
	rnn_ppg.x_data, rnn_ppg.y_data, rnn_ppg.subj_inputs = rnn_ppg.create_dataset(rnn_ppg.window, rnn_ppg.overlap, rnn_ppg.decimation)
	x_test, y_test = rnn_ppg.partition_data(rnn_ppg.subjs_test)
	np.savetxt(f'x_test_{rnn_ppg.window:04d}.csv', x_test.reshape((len(x_test), -1)), delimiter = ',')
	np.savetxt(f'y_test_{rnn_ppg.window:04d}.csv', tf.one_hot(y_test.reshape((len(y_test),)), 3).numpy(), delimiter = ',')
