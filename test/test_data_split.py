import os

import numpy as np
import pickle

DIR = os.getcwd()

with open(os.path.join(DIR, 'labeled_dump_STEAD', 'window_6000'), 'rb') as f:
  data = pickle.load(f)

import random
random.shuffle(data)

waves = 2000
X = []
y = []
X_valid =[]
y_valid = []
for i in range(waves):
  X.append(data[i]['data'])
  y.append(data[i]['p_time_label'])

test_x = np.asarray(X)
tesy_x = test_x.reshape(waves,6000,3)
test_y = np.asarray(y)
test_y = test_y.reshape(waves,6000,1)


X_train, y_train = test_x[:int(waves*0.7)], test_y[:int(waves*0.7)]
X_valid, y_valid = test_x[int(waves*0.7):int(waves*(1-0.1))], test_y[int(waves*0.7):int(waves*(1-0.1))]
X_test, y_test = test_x[int(waves*(1-0.1)):], test_y[int(waves*(1-0.1)):]

for name, var in [['Sample_X_train', X_train], ['Sample_p_train', y_train], ['Sample_X_val', X_valid],
                  ['Sample_p_val', y_valid],['Sample_X_test', X_test], ['Sample_y_test', y_test]]:
  with open(os.path.join(DIR, 'test_6000_data', name), 'wb') as f:
    pickle.dump(var, f)