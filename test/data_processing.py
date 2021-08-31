import os
import random

import pickle

## pickle file 읽어오기
import numpy as np

def data_processing() :

    DIR = os.getcwd()
    with open(os.path.join(DIR, 'data', 'test'), 'rb') as f:
      data = pickle.load(f)

    random.shuffle(data)

    DIR = os.getcwd()
    print(data[0]['p_label'])
    waves = 2000
    X = []
    y = []
    X_valid =[]
    y_valid = []
    for i in range(waves):
      X.append(data[i]['data'])
      y.append(data[i]['p_time_label'])


    test_x = np.asarray(X)
    test_x = test_x.reshape(waves,6000,3)
    test_y = np.asarray(y)
    test_y = test_y.reshape(waves,6)


    X_train, y_train = test_x[:int(waves*0.7)], test_y[:int(waves*0.7)]
    X_valid, y_valid = test_x[int(waves*0.7):int(waves*(1-0.1))], test_y[int(waves*0.7):int(waves*(1-0.1))]
    X_test, y_test = test_x[int(waves*(1-0.1)):], test_y[int(waves*(1-0.1)):]

    return X_train, y_train, X_valid, y_valid, X_test, y_test