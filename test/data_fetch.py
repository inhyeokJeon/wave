import os
import pickle
import random

def data_fetch(data_path):
    DIR = os.getcwd()

    data_list =['Sample_X_train', 'Sample_p_train',
                'Sample_X_val','Sample_p_val']
    '''
    
    with open(os.path.join(DIR, data_path, 'window_6000'), 'rb') as f:
        data = pickle.load(f)
    random.shuffle(data)

    '''
    result_list =[]
    for name in data_list:
        with open(os.path.join(DIR, data_path, name), 'rb') as f:
            data = pickle.load(f)
        result_list.append(data)

    X_train, y_train, X_valid, y_valid \
        = result_list[0],result_list[1],result_list[2],result_list[3]

    return X_train, y_train, X_valid, y_valid

def test_fetch(data_path):
    print("1")