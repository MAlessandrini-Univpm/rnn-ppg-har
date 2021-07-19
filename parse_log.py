import sys
import ast


fit_accuracy = []  # list of lists
fit_val_accuracy = []  # list of lists
test_accuracy = []  # list of numbers
oversample = False


with open(sys.argv[1], 'r') as f:
	for line in f:
		if line.startswith('fit_accuracy'):
			fit_accuracy.append(ast.literal_eval(line[13:]))
		elif line.startswith('fit_val_accuracy'):
			fit_val_accuracy.append(ast.literal_eval(line[17:]))
		elif line.startswith('test_accuracy'):
			test_accuracy.append(float(line[13:]))
		elif line.startswith('window'):
			window = line[7:].strip()
		elif line.startswith('layers'):
			layers = line[7:].strip()
		elif line.startswith('oversample'):
			oversample = ast.literal_eval(line[11:])

n = len(fit_accuracy)
assert((n == len(fit_val_accuracy) or not fit_val_accuracy) and n == len(test_accuracy))

a_trn = [0., 0.]  # average of average, average of max
a_val = [0., 0.]  # average of average, average of max
a_tst = [0., 0.]  # average, max

a_trn[0] = sum([sum(l) / len(l) for l in fit_accuracy]) / n
a_trn[1] = sum([max(l) for l in fit_accuracy]) / n
a_val[0] = sum([sum(l) / len(l) for l in fit_val_accuracy]) / n
a_val[1] = sum([max(l) for l in fit_val_accuracy]) / n
a_tst[0] = sum(test_accuracy) / n
a_tst[1] = max(test_accuracy)

print(sys.argv[1], layers, window, 'OVR' if oversample else '')
print(f'[{n:2d}]     avg      max')
print(f'train {a_trn[0]:7.4f}  {a_trn[1]:7.4f}')
print(f'val   {a_val[0]:7.4f}  {a_val[1]:7.4f}')
print(f'test  {a_tst[0]:7.4f}  {a_tst[1]:7.4f}')
