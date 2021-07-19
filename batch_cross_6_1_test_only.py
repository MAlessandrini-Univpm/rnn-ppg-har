import rnn_ppg
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import re

# Note: we create global variables in the rnn_ppg module, because functions there expect that

rnn_ppg.dense1 = 32
rnn_ppg.lstm1 = 32
rnn_ppg.lstm2 = 32
rnn_ppg.lstm3 = 32

#rnn_ppg.window = 1200
#rnn_ppg.overlap = 1200 // 2

#rnn_ppg.subjs_train_perm = ( ((0, 1, 2, 3), (4,)), ((0, 1, 2, 4), (3,)), ((0, 1, 3, 4), (2,)), ((0, 2, 3, 4), (1,)), ((1, 2, 3, 4), (0,)))  # cross-validation
#rnn_ppg.subjs_train_perm = ( ((0, 1, 2, 3, 4), ()), )  # final model
#rnn_ppg.subjs_test = (5, 6)
rnn_ppg.epochs = 100
rnn_ppg.iterations = 1

rnn_ppg.oversample = False
#rnn_ppg.decimation = 2


model_list = sorted(sys.argv[1:])
assert(len(model_list) == 7)
for i in range(0, 7):
	assert(model_list[i].startswith(f'out{i}'))
# find decimation from file names
dec = None
expr = re.compile('out[0-6].*d([0-9]+).*\\.h5')
for i in range(0, 7):
	m = expr.fullmatch(model_list[i])
	assert(m is not None)
	if dec is None:
		dec = int(m.group(1))
	else:
		assert(int(m.group(1)) == dec)

print('### using decimation', dec)

for window in (1200, ):
	for ovr in (2, ):
		for decimation in (dec,):
			rnn_ppg.window = window
			rnn_ppg.overlap = window // ovr if ovr else 0
			rnn_ppg.decimation = decimation
			if rnn_ppg.decimation:
				rnn_ppg.window //= rnn_ppg.decimation
				rnn_ppg.overlap //= rnn_ppg.decimation
			rnn_ppg.x_data, rnn_ppg.y_data, rnn_ppg.subj_inputs = rnn_ppg.create_dataset(rnn_ppg.window, rnn_ppg.overlap, rnn_ppg.decimation)
			if rnn_ppg.oversample:
				rnn_ppg.x_data, rnn_ppg.y_data, rnn_ppg.subj_inputs = rnn_ppg.oversampling(rnn_ppg.x_data, rnn_ppg.y_data, rnn_ppg.subj_inputs, 7)
			conf_tot = np.zeros((3,3))
			for leftout in range(0, 7):
				rnn_ppg.subjs_train_perm = ( ( tuple(x for x in (0, 1, 2, 3, 4, 5, 6) if x != leftout) , ()), )  # not used in testing
				rnn_ppg.subjs_test = (leftout,)
				rnn_ppg.train_session(load_model = model_list[leftout])
				# confusion matrix
				x_data_test, y_data_test = rnn_ppg.partition_data(rnn_ppg.subjs_test)
				model = keras.models.load_model(model_list[leftout])
				y_pred = model.predict_classes(x_data_test)
				con_mat = tf.math.confusion_matrix(labels = y_data_test, predictions = y_pred).numpy()
				print(con_mat)
				conf_tot += con_mat
			print('\ntotal')
			print(conf_tot)
			#print(np.around(conf_tot.astype('float') / con_mat.sum(axis = 1)[:, np.newaxis], decimals = 2))
			print(conf_tot / np.sum(conf_tot, 1)[:, None])
			print(np.trace(conf_tot) / np.sum(conf_tot))

