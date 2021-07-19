import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from sklearn.preprocessing import StandardScaler
import scipy.signal


# read all the dataset in memory
dataset_dir = './PPG_ACC_dataset'
subjects = (1, 2, 3, 4, 5, 6, 7)
activities = ('rest', 'squat', 'step')
dataset = {}  # { subject → { category → data[5 * [4,n] }}


def test_data(fusion):
	# columns = acc, acc, acc, ppg
	if np.isnan(fusion).sum():
		for col in range(0, 4):
			max_seq, seq = 0, 0
			for n in fusion[:, col]:
				if np.isnan(n):
					seq += 1
					max_seq = max(max_seq, seq)
				else:
					seq = 0
			if max_seq: print('max NaN seq:', max_seq)
	# zero holes
	for col in range(3, 4):
		max_seq, seq = 0, 0
		for n in fusion[:, col]:
			if not n:
				seq += 1
				max_seq = max(max_seq, seq)
			else:
				seq = 0
		if max_seq: print('max zero seq:', max_seq)


def clean_data(fusion, name):
	# some tracks have isolated NaNs
	for col in range(0, 4):
		ids = np.where(np.isnan(fusion[:, col]))[0]
		#if ids.size: print(name, 'Fixing NaNs in ACC and/or PPG')
		for row in ids:
			fusion[row, col] = 0.5 * (fusion[row - 1, col] + fusion[row + 1, col])
	# some PPG tracks have periodic, isolated zeros, resulting in spikes
	for col in range(3, 4):
		ids = np.where(fusion[:, col] == 0)[0]
		#if ids.size: print(name, 'Fixing zeros in PPG')
		for row in ids:
			fusion[row, col] = 0.5 * (fusion[row - 1, col] + fusion[row + 1, col])
	# many acc. tracks have periodic, single-point spikes of no specific values. Let's test all of them, even if it's slow
	for col in range(0, 3):
		found = False
		for row in range(1, len(fusion) - 1):
			if abs(fusion[row,col] - fusion[row-1,col]) > 5000 and abs(fusion[row,col] - fusion[row+1,col]) > 5000:
				found = True
				fusion[row, col] = 0.5 * (fusion[row - 1, col] + fusion[row + 1, col])
		#if found: print(name, 'Fixing spikes in ACC')


for subject in subjects:
	dataset[subject] = {}
	for category, name in enumerate(activities):
		data = []
		for record in range(0, 5):
			acc = scipy.io.loadmat(dataset_dir + f'/S{subject}/{name}{record + 1}_acc.mat')['ACC']
			ppg = scipy.io.loadmat(dataset_dir + f'/S{subject}/{name}{record + 1}_ppg.mat')['PPG'][:, 0:2]  # some PPG files have 3 columns instead of 2
			fusion = np.hstack((acc[:, 1:], ppg[:, 1:]))  # remove x axis (time)
			clean_data(fusion, f'S{subject}/{name}{record + 1}')
			fusion[:, 0:3] /= 16384.
			fusion[:, 3] /= 8.  # nA
			#print(np.std(fusion[:,0]), np.std(fusion[:,1]), np.std(fusion[:,2]))
			data.append(fusion)
		dataset[subject][category] = data


colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

for cat in (0, ):
	plt.figure()
	for subj in (2, ):
		#for cat, data in dataset[1].items():
		data = dataset[subj][cat]
		for rec in (0, 1, 2, 3, 4):
			for channel in (2,):
				plot_data = np.transpose(data[rec])[channel]
				# apply standardization to single windows
				#w = 0
				#while w < len(plot_data):  # standardize for every window
				#	w_len = min(1200, len(plot_data) - w)
				#	plot_data[w:w+w_len] = StandardScaler().fit_transform(plot_data[w:w+w_len].reshape(-1, 1)).reshape((w_len, ))
				#	w += 1200
				#
				plt.plot(np.arange(0, 0.0025 * len(plot_data), 0.0025), plot_data, colors[subj - 1])
				#undersampled = scipy.signal.decimate(plot_data, 2)
				#plt.plot(np.arange(0, 0.005 * len(undersampled), 0.005), undersampled, 'b')
				#np.savetxt(f'subj{subj}_{activities[cat]}_ch{channel}_rec{rec}', plot_data)
	plt.title(f'activity: {activities[cat]}')
plt.show()
