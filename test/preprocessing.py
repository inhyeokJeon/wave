#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import shutil

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

DIR = os.getcwd()

data_path_chunk1 = os.path.join(DIR, 'data/chunk1')
data_path_chunk2 = os.path.join(DIR, 'data/chunk2')


def preprocessing(params):
    '''
    params -> dict,
    window_count -> int
    file_name -> str
    imbalance ->  bool
    '''

    window_size = 6000 // params['window_count']
    print(os.path.join(DIR, 'labeled_dump_STEAD', params['file_name']))
    print(os.path.isfile(os.path.join(DIR, 'labeled_dump_STEAD', params['file_name'])))

    if os.path.isfile(os.path.join(DIR, 'labeled_dump_STEAD', params['file_name'])):
        print('file exist!!!')
        return
    else :
        file_name = "chunk2.hdf5"
        csv_file = "chunk2.csv"

        # reading the csv file into a dataframe:
        df = pd.read_csv(os.path.join(data_path_chunk2, csv_file))
        print(f'total events in csv file: {len(df)}')

        # filterering the dataframe
        # df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
        '''
        if params['condition'][1] == '<':
            df = df[(df.source_magnitude < int(params['condition'][2]))]
        else:
            df = df[(df.source_magnitude > int(params['condition'][2]))]
        '''
        print(f'total events selected: {len(df)}')

        # making a list of trace names for the selected data
        random_idx = np.random.choice(len(df), 15000)
        ev_list = df['trace_name'].to_numpy()[random_idx]
        print('random choice length :', len(ev_list))

        # retrieving selected waveforms from the hdf5 file:
        dtfl = h5py.File(os.path.join(data_path_chunk2, file_name), 'r')

        outputs = list()

        for c, evi in tqdm(enumerate(ev_list)):
            dataset = dtfl.get('data/' + str(evi))

            data = np.array(dataset)
            p_time_label = np.zeros(params['window_count'])
            s_time_label = np.zeros(params['window_count'])

            p_time_label[int(dataset.attrs['p_arrival_sample'] // window_size)] = 1
            s_time_label[int(dataset.attrs['s_arrival_sample'] // window_size)] = 1

            outputs.append({
                'data': data,
                'p_label': 1,
                's_label': 1,
                'p_time_label': p_time_label,
                's_time_label': s_time_label

            })

        file_name = "chunk1.hdf5"
        csv_file = "chunk1.csv"

        # reading the csv file into a dataframe:
        df = pd.read_csv(os.path.join(data_path_chunk1, csv_file))
        print(f'total events in csv file: {len(df)}')

        # filterering the dataframe
        df = df[(df.trace_category == 'noise')]
        print(f'total events selected: {len(df)}')

        # making a list of trace names for the selected data
        random_idx = np.random.choice(len(ev_list), 5000)
        ev_list = df['trace_name'].to_numpy()[random_idx]
        print('random choice length :', len(ev_list))

        # retrieving selected waveforms from the hdf5 file:
        dtfl = h5py.File(os.path.join(data_path_chunk1, file_name), 'r')

        for c, evi in tqdm(enumerate(ev_list)):
            dataset = dtfl.get('data/' + str(evi))

            data = np.array(dataset)
            p_time_label = np.zeros(params['window_count'])
            s_time_label = np.zeros(params['window_count'])

            outputs.append({
                'data': data,
                'p_label': 0,
                's_label': 0,
                'p_time_label': p_time_label,
                's_time_label': s_time_label
            })
        print("hello")
        with open(os.path.join(DIR, 'labeled_dump_STEAD', params['file_name']), 'wb') as f:
            pickle.dump(outputs, f)







