import rnn_ppg

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

rnn_ppg.oversample = True  # set to False for cross-validation
#rnn_ppg.decimation = 2

for window in (1200, ):
	for ovr in (2, ):
		for decimation in (150,100,80,60,50):
			for leftout in range(0, 7):
				rnn_ppg.subjs_train_perm = ( ( tuple(x for x in (0, 1, 2, 3, 4, 5, 6) if x != leftout) , ()), )
				rnn_ppg.subjs_test = (leftout,)
				rnn_ppg.window = window
				rnn_ppg.overlap = window // ovr if ovr else 0
				rnn_ppg.decimation = decimation
				if rnn_ppg.decimation:
					rnn_ppg.window //= rnn_ppg.decimation
					rnn_ppg.overlap //= rnn_ppg.decimation
				rnn_ppg.x_data, rnn_ppg.y_data, rnn_ppg.subj_inputs = rnn_ppg.create_dataset(rnn_ppg.window, rnn_ppg.overlap, rnn_ppg.decimation)
				if rnn_ppg.oversample:
					rnn_ppg.x_data, rnn_ppg.y_data, rnn_ppg.subj_inputs = rnn_ppg.oversampling(rnn_ppg.x_data, rnn_ppg.y_data, rnn_ppg.subj_inputs, 7)
				rnn_ppg.train_session(save_model = True, file_id = f'out{leftout}_')
