import os
import pickle


def data_fetch(data_path):
    DIR = os.getcwd()

    data_list =['Sample_X_train', 'Sample_p_train',
                'Sample_X_val','Sample_p_val',
                'Sample_X_test','Sample_p_test']

    result_list =[]
    for name in data_list:
        with open(os.path.join(DIR, data_path, name), 'rb') as f:
            data = pickle.load(f)
            result_list.append(data)

    X_train, y_train, X_valid, y_valid, X_test, y_test \
        = result_list[0],result_list[1],result_list[2],result_list[3],result_list[4],result_list[5]

    return X_train, y_train, X_valid, y_valid, X_test, y_test