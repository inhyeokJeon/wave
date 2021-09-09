import os
import random

import pickle

## pickle file 읽어오기
import shutil

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def _split(args, save_dir):
    """

    Split the list of input data into training, validation, and test set.
    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.

    save_dir: str
       Path to the output directory.

    Returns
    -------
    training: str
        List of trace names for the training set.
    validation : str
        List of trace names for the validation set.

    """

    '''
    # 원래꺼
    df = pd.read_csv(args['input_csv'])
    
    ev_list = df.trace_name.tolist()
    np.random.shuffle(ev_list)
    
    '''
    # chunk1,2 합친것에서 noise 2000 개 local 10000개 뽑아서, 섞고 pickle 로 저장했음.
    df = pd.read_csv(args['input_csv'])
    df_local = df[(df.trace_category == 'earthquake_local')]
    df_noise = df[(df.trace_category == 'noise')]

    random_idx = np.random.choice(len(df_local), 10000)
    ev_list_local = df_local['trace_name'].to_numpy()[random_idx]
    random_idx = np.random.choice(len(df_noise), 2000)
    ev_list_noise = df_noise['trace_name'].to_numpy()[random_idx]
    ev_list = np.concatenate((ev_list_local, ev_list_noise), axis=0)

    np.random.shuffle(ev_list)

    training = ev_list[:int(args['train_valid_test_split'][0] * len(ev_list))]
    validation = ev_list[int(args['train_valid_test_split'][0] * len(ev_list)):
                         int(args['train_valid_test_split'][0] * len(ev_list) + args['train_valid_test_split'][1] * len(
                             ev_list))]
    test = ev_list[
           int(args['train_valid_test_split'][0] * len(ev_list) + args['train_valid_test_split'][1] * len(ev_list)):]
    # todo

    np.save(save_dir + '/test', test)
    return training, validation

def _make_dir(output_name):
    """

    Make the output directories.
    Parameters
    ----------
    output_name: str
        Name of the output directory.

    Returns
    -------
    save_dir: str
        Full path to the output directory.

    save_models: str
        Full path to the model directory.

    """

    if output_name == None:
        print('Please specify output_name!')
        return
    else:
        save_dir = os.path.join(os.getcwd(), str(output_name) + '_outputs')
        save_models = os.path.join(save_dir, 'models')
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_models)
    return save_dir, save_models

def data_reader(list_IDs,
                file_name,
                dim=6000,
                n_channels=3,
                norm_mode='max',
                augmentation=False,
                add_event_r=None,
                add_gap_r=None,
                coda_ratio=0.4,
                shift_event_r=None,
                add_noise_r=None,
                drop_channe_r=None,
                scale_amplitude_r=None,
                pre_emphasis=True):
    """

    For pre-processing and loading of data into memory.

    Parameters
    ----------
    list_IDsx: str
        List of trace names.

    file_name: str
        Path to the input hdf5 datasets.

    dim: int, default=6000
        Dimension of input traces, in sample.

    n_channels: int, default=3
        Number of channels.

    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.

    augmentation: bool, default=True
        If True, half of each batch will be augmented version of the other half.

    add_event_r: {float, None}, default=None
        Chance for randomly adding a second event into the waveform.
    add_gap_r: {float, None}, default=None
        Add an interval with zeros into the waveform representing filled gaps.
    coda_ratio: {float, 0.4}, default=0.4
        % of S-P time to extend event/coda envelope past S pick.

    shift_event_r: {float, None}, default=0.9
        Rate of augmentation for randomly shifting the event within a trace.

    add_noise_r: {float, None}, default=None
        Chance for randomly adding Gaussian noise into the waveform.

    drop_channe_r: {float, None}, default=None
        Chance for randomly dropping some of the channels.

    scale_amplitude_r: {float, None}, default=None
        Chance for randomly amplifying the waveform amplitude.

    pre_emphasis: bool, default=False
        If True, waveforms will be pre emphasized.
    Returns
    --------
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.

    Note
    -----
    Label type is fixed to box.


    """

    def _normalize(data, mode='max'):
        'Normalize waveforms in each batch'

        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert (max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data

        elif mode == 'std':
            std_data = np.std(data, axis=0, keepdims=True)
            assert (std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data

    def _scale_amplitude(data, rate):
        'Scale amplitude or waveforms'

        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2 * rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel(data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10):
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(data, rate):
        'Randomly replace values of one or two components to zeros in noise data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate:
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(data, rate):
        'Randomly add gaps (zeros) of different sizes into waveforms'

        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate:
            data[gap_start:gap_end, :] = 0
        return data

    def _add_noise(data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'

        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0):
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:, 0] + np.random.normal(0, np.random.uniform(0.01, 0.15) * max(data[:, 0]),
                                                             data.shape[0])
            data_noisy[:, 1] = data[:, 1] + np.random.normal(0, np.random.uniform(0.01, 0.15) * max(data[:, 1]),
                                                             data.shape[0])
            data_noisy[:, 2] = data[:, 2] + np.random.normal(0, np.random.uniform(0.01, 0.15) * max(data[:, 2]),
                                                             data.shape[0])
        else:
            data_noisy = data
        return data_noisy

    def _adjust_amplitude_for_multichannels(data):
        'Adjust the amplitude of multichaneel data'

        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert (tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
            data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(a=0, b=20, c=40):
        'Used for triangolar labeling'

        z = np.linspace(a, c, num=2 * (b - a) + 1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half] - a) / (b - a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c - z[second_half]) / (c - b)
        return y

    def _add_event(data, addp, adds, coda_end, snr, rate):
        'Add a scaled version of the event into the empty part of the trace'

        added = np.copy(data)
        additions = spt_secondEV = sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr >= 10.0) and (data.shape[0] - s_p - 21 - coda_end) > 20:
                secondEV_strt = np.random.randint(coda_end, data.shape[0] - s_p - 21)
                scaleAM = 1 / np.random.randint(1, 10)
                space = data.shape[0] - secondEV_strt
                added[secondEV_strt:secondEV_strt + space, 0] += data[addp:addp + space, 0] * scaleAM
                added[secondEV_strt:secondEV_strt + space, 1] += data[addp:addp + space, 1] * scaleAM
                added[secondEV_strt:secondEV_strt + space, 2] += data[addp:addp + space, 2] * scaleAM
                spt_secondEV = secondEV_strt
                if spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:
                    additions = [spt_secondEV, sst_secondEV]
                    data = added
        return data, additions

    def _shift_event(data, addp, adds, coda_end, snr, rate):
        'Randomly rotate the array to shift the event location'

        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]

            if addp + nrotate >= 0 and addp + nrotate < org_len:
                addp2 = addp + nrotate;
            else:
                addp2 = None;
            if adds + nrotate >= 0 and adds + nrotate < org_len:
                adds2 = adds + nrotate;
            else:
                adds2 = None;
            if coda_end + nrotate < org_len:
                coda_end2 = coda_end + nrotate
            else:
                coda_end2 = org_len
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end = coda_end2;
        return data, addp, adds, coda_end

    def _pre_emphasis(data, pre_emphasis=0.97):
        'apply the pre_emphasis'

        for ch in range(n_channels):
            bpf = data[:, ch]
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data

    fl = h5py.File(file_name, 'r')

    if augmentation:
        X = np.zeros((2 * len(list_IDs), dim, n_channels))
        y1 = np.zeros((2 * len(list_IDs), dim, 1))
        y2 = np.zeros((2 * len(list_IDs), dim, 1))
        y3 = np.zeros((2 * len(list_IDs), dim, 1))
    else:
        X = np.zeros((len(list_IDs), dim, n_channels))
        y1 = np.zeros((len(list_IDs), dim, 1))
        y2 = np.zeros((len(list_IDs), dim, 1))
        y3 = np.zeros((len(list_IDs), dim, 1))

        # Generate data
    pbar = tqdm(total=len(list_IDs))
    for i, ID in enumerate(list_IDs):
        pbar.update()

        additions = None
        dataset = fl.get('data/' + str(ID))

        if ID.split('_')[-1] == 'EV':
            data = np.array(dataset)
            spt = int(dataset.attrs['p_arrival_sample']);
            sst = int(dataset.attrs['s_arrival_sample']);
            coda_end = int(dataset.attrs['coda_end_sample']);
            snr = dataset.attrs['snr_db'];

        elif ID.split('_')[-1] == 'NO':
            data = np.array(dataset)

        if augmentation:
            if dataset.attrs['trace_category'] == 'earthquake_local':
                data, spt, sst, coda_end = _shift_event(data, spt, sst, coda_end, snr, shift_event_r / 2);
            if norm_mode:
                data1 = _normalize(data, norm_mode)

            if dataset.attrs['trace_category'] == 'earthquake_local':
                if shift_event_r and spt:
                    data, spt, sst, coda_end = _shift_event(data, spt, sst, coda_end, snr, shift_event_r);

                if add_event_r:
                    data, additions = _add_event(data, spt, sst, coda_end, snr, add_event_r);

                if drop_channe_r:
                    data = _drop_channel(data, snr, drop_channe_r);
                #  data = _adjust_amplitude_for_multichannels(data);

                if scale_amplitude_r:
                    data = _scale_amplitude(data, scale_amplitude_r);

                if pre_emphasis:
                    data = _pre_emphasis(data);

                if add_noise_r:
                    data = _add_noise(data, snr, add_noise_r);

                if norm_mode:
                    data2 = _normalize(data, norm_mode);

            if dataset.attrs['trace_category'] == 'noise':
                if drop_channe_r:
                    data = _drop_channel_noise(data, drop_channe_r);
                if add_gap_r:
                    data = _add_gaps(data, add_gap_r)
                if norm_mode:
                    data2 = _normalize(data, norm_mode)

            X[i, :, :] = data1
            X[len(list_IDs) + i, :, :] = data2

            if dataset.attrs['trace_category'] == 'earthquake_local':
                if spt and (spt - 20 >= 0) and (spt + 21 < dim):
                    y2[i, spt - 20:spt + 21, 0] = _label()
                    y2[len(list_IDs) + i, spt - 20:spt + 21, 0] = _label()
                elif spt and (spt + 21 < dim):
                    y2[i, 0:spt + spt + 1, 0] = _label(a=0, b=spt, c=2 * spt)
                    y2[len(list_IDs) + i, 0:spt + spt + 1, 0] = _label(a=0, b=spt, c=2 * spt)
                elif spt and (spt - 20 >= 0):
                    pdif = dim - spt
                    y2[i, spt - pdif - 1:dim, 0] = _label(a=spt - pdif, b=spt, c=2 * pdif)
                    y2[len(list_IDs) + i, spt - pdif - 1:dim, 0] = _label(a=spt - pdif, b=spt, c=2 * pdif)

                if sst and (sst - 20 >= 0) and (sst + 21 < dim):
                    y3[i, sst - 20:sst + 21, 0] = _label()
                    y3[len(list_IDs) + i, sst - 20:sst + 21, 0] = _label()
                elif sst and (sst + 21 < dim):
                    y3[i, 0:sst + sst + 1, 0] = _label(a=0, b=sst, c=2 * sst)
                    y3[len(list_IDs) + i, 0:sst + sst + 1, 0] = _label(a=0, b=sst, c=2 * sst)
                elif sst and (sst - 20 >= 0):
                    sdif = dim - sst
                    y3[i, sst - sdif - 1:dim, 0] = _label(a=sst - sdif, b=sst, c=2 * sdif)
                    y3[len(list_IDs) + i, sst - sdif - 1:dim, 0] = _label(a=sst - sdif, b=sst, c=2 * sdif)

                sd = sst - spt
                if sst + int(coda_ratio * sd) <= dim:
                    y1[i, spt:int(sst + (coda_ratio * sd)), 0] = 1
                    y1[len(list_IDs) + i, spt:int(sst + (coda_ratio * sd)), 0] = 1
                else:
                    y1[i, spt:dim, 0] = 1
                    y1[len(list_IDs) + i, spt:dim, 0] = 1

                if additions:
                    add_spt = additions[0];
                    print(add_spt)
                    add_sst = additions[1];
                    add_sd = add_sst - add_spt

                    if add_spt and (add_spt - 20 >= 0) and (add_spt + 21 < dim):
                        y2[len(list_IDs) + i, add_spt - 20:add_spt + 21, 0] = _label()
                    elif add_spt and (add_spt + 21 < dim):
                        y2[len(list_IDs) + i, 0:add_spt + add_spt + 1, 0] = _label(a=0, b=add_spt, c=2 * add_spt)
                    elif add_spt and (add_spt - 20 >= 0):
                        pdif = dim - add_spt
                        y2[len(list_IDs) + i, add_spt - pdif - 1:dim, 0] = _label(a=add_spt - pdif, b=add_spt,
                                                                                  c=2 * pdif)

                    if add_sst and (add_sst - 20 >= 0) and (add_sst + 21 < dim):
                        y3[len(list_IDs) + i, add_sst - 20:add_sst + 21, 0] = _label()
                    elif add_sst and (add_sst + 21 < dim):
                        y3[len(list_IDs) + i, 0:add_sst + add_sst + 1, 0] = _label(a=0, b=add_sst, c=2 * add_sst)
                    elif add_sst and (add_sst - 20 >= 0):
                        sdif = dim - add_sst
                        y3[len(list_IDs) + i, add_sst - sdif - 1:dim, 0] = _label(a=add_sst - sdif, b=add_sst,
                                                                                  c=2 * sdif)

                    if add_sst + int(coda_ratio * add_sd) <= dim:
                        y1[len(list_IDs) + i, add_spt:int(add_sst + (coda_ratio * add_sd)), 0] = 1
                    else:
                        y1[len(list_IDs) + i, add_spt:dim, 0] = 1

    fl.close()
    # todo
    return X.astype('float32'), y2.astype('float32')
    #return X.astype('float32'), y1.astype('float32'), y2.astype('float32'), y3.astype('float32')

#-----------------------------------------------------------#
#test#

def main():
    input_dimention = (6000, 3)
    cnn_blocks = 5
    lstm_blocks = 2
    padding = 'same'
    activation = 'relu'
    drop_rate = 0.1
    shuffle = True
    label_type = 'gaussian'
    normalization_mode = 'std'
    augmentation = False
    add_event_r = 0.6
    shift_event_r = 0.99
    add_noise_r = 0.3
    drop_channel_r = 0.5
    add_gap_r = 0.2
    coda_ratio = 0.4
    scale_amplitude_r = None
    pre_emphasis = False
    loss_weights = [0.05, 0.40, 0.55]
    loss_types = ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
    train_valid_test_split = [0.85, 0.05, 0.10]
    mode = 'generator'
    batch_size = 200
    epochs = 200
    monitor = 'val_loss'
    patience = 12
    gpuid = None
    gpu_limit = None
    use_multiprocessing = True

    args = {
        "input_hdf5": 'data/merged.hdf5',
        "input_csv": 'data/merged.csv',
        "output_name": 'test_trainer',
        "input_dimention": input_dimention,
        "cnn_blocks": cnn_blocks,
        "lstm_blocks": lstm_blocks,
        "padding": padding,
        "activation": activation,
        "drop_rate": drop_rate,
        "shuffle": shuffle,
        "label_type": label_type,
        "normalization_mode": normalization_mode,
        "augmentation": augmentation,
        "add_event_r": add_event_r,
        "shift_event_r": shift_event_r,
        "add_noise_r": add_noise_r,
        "add_gap_r": add_gap_r,
        "coda_ratio": coda_ratio,
        "drop_channel_r": drop_channel_r,
        "scale_amplitude_r": scale_amplitude_r,
        "pre_emphasis": pre_emphasis,
        "loss_weights": loss_weights,
        "loss_types": loss_types,
        "train_valid_test_split": train_valid_test_split,
        "mode": mode,
        "batch_size": batch_size,
        "epochs": epochs,
        "monitor": monitor,
        "patience": patience,
        "gpuid": gpuid,
        "gpu_limit": gpu_limit,
        "use_multiprocessing": use_multiprocessing
    }
    print("HH")
    save_dir, save_models = _make_dir(args['output_name'])
    training, validation = _split(args, save_dir)
    print(len(training))

    X_train, y2_train = data_reader(list_IDs=training,
                                file_name=str(args['input_hdf5']),
                                dim=args['input_dimention'][0],
                                n_channels=args['input_dimention'][-1],
                                norm_mode=args['normalization_mode'],
                                augmentation=args['augmentation'],
                                add_event_r=args['add_event_r'],
                                add_gap_r=args['add_gap_r'],
                                coda_ratio=args['coda_ratio'],
                                shift_event_r=args['shift_event_r'],
                                add_noise_r=args['add_noise_r'],
                                drop_channe_r=args['drop_channel_r'],
                                scale_amplitude_r=args['scale_amplitude_r'],
                                pre_emphasis=args['pre_emphasis'])

    X_val, y2_val = data_reader(list_IDs=validation,
                                    file_name=str(args['input_hdf5']),
                                    dim=args['input_dimention'][0],
                                    n_channels=args['input_dimention'][-1],
                                    norm_mode=args['normalization_mode'],
                                    augmentation=args['augmentation'],
                                    add_event_r=args['add_event_r'],
                                    add_gap_r=args['add_gap_r'],
                                    coda_ratio=args['coda_ratio'],
                                    shift_event_r=args['shift_event_r'],
                                    add_noise_r=args['add_noise_r'],
                                    drop_channe_r=args['drop_channel_r'],
                                    scale_amplitude_r=args['scale_amplitude_r'],
                                    pre_emphasis=args['pre_emphasis'])
    '''
    X_test, y2_test = data_reader(list_IDs=test,
                                file_name=str(args['input_hdf5']),
                                dim=args['input_dimention'][0],
                                n_channels=args['input_dimention'][-1],
                                norm_mode=args['normalization_mode'],
                                augmentation=args['augmentation'],
                                add_event_r=args['add_event_r'],
                                add_gap_r=args['add_gap_r'],
                                coda_ratio=args['coda_ratio'],
                                shift_event_r=args['shift_event_r'],
                                add_noise_r=args['add_noise_r'],
                                drop_channe_r=args['drop_channel_r'],
                                scale_amplitude_r=args['scale_amplitude_r'],
                                pre_emphasis=args['pre_emphasis'])

    '''

    DIR = os.getcwd()
    print(len(X_train))
    for name, var in [['Sample_X_train', X_train], ['Sample_p_train',y2_train],['Sample_X_val',X_val],['Sample_p_val',y2_val]]:
        #['Sample_X_test', X_test], ['Sample_p_test',y2_test]:
        with open(os.path.join(DIR, 'test_STEAD', name), 'wb') as f:
            pickle.dump(var, f)

    '''
    params = {
        'window_count' : 3000,
        'file_name' : "window_3000"
    }

    preprocessing(params)
    '''


if __name__ == "__main__":
    main()