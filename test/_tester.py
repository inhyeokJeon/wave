
from __future__ import print_function

import os
from obspy.signal.trigger import trigger_onset
from tensorflow import keras

os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import h5py
import time
import shutil
from utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
np.warnings.filterwarnings('ignore')
import datetime
from tqdm import tqdm
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):
    """

    Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.

    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.

    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).

    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.

    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).

    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.

    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.
    Returns
    ---------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Modified from
   ----------------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def generate_arrays_from_file(file_list, step):
    """

    Make a generator to generate list of trace names.

    Parameters
    ----------
    file_list : str
        A list of trace names.

    step : int
        Batch size.

    Returns
    --------
    chunck : str
        A batch of trace names.

    """

    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i * step + step
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b = e
            yield chunck

class PreLoadGeneratorTest(keras.utils.Sequence):
    """

    Keras generator with preprocessing. For testing. Pre-load version.

    Parameters
    ----------
    list_IDsx: str
        List of trace names.

    file_name: str
        Path to the input hdf5 file.

    dim: tuple
        Dimension of input traces.

    batch_size: int, default=32.
        Batch size.

    n_channels: int, default=3.
        Number of channels.

    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'

    Returns
    --------
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.


    """

    def __init__(self,
                 list_IDs,
                 inp_data,
                 dim,
                 batch_size=32,
                 n_channels=3,
                 norm_mode='std'):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.inp_data = inp_data
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def _normalize(self, data, mode='max'):
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

    def __data_generation(self, list_IDs_temp):
        'readint the waveforms'
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            dataset = self.inp_data[ID]
            data = np.array(dataset)
            data = self._normalize(data, self.norm_mode)
            X[i, :, :] = data

        return X

def picker(args, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, spt=None, sst=None):
    """

    Performs detection and picking.
    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.

    yh1 : 1D array
        Detection probabilities.

    yh2 : 1D array
        P arrival probabilities.

    yh3 : 1D array
        S arrival probabilities.

    yh1_std : 1D array
        Detection standard deviations.

    yh2_std : 1D array
        P arrival standard deviations.

    yh3_std : 1D array
        S arrival standard deviations.

    spt : {int, None}, default=None
        P arrival time in sample.

    sst : {int, None}, default=None
        S arrival time in sample.


    Returns
    --------
    matches: dic
        Contains the information for the detected and picked event.

    matches: dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}

    pick_errors : dic
        {detection statr-time:[ P_ground_truth - P_pick, S_ground_truth - S_pick]}

    yh3: 1D array
        normalized S_probability

    """

    #   yh3[yh3>0.04] = ((yh1+yh3)/2)[yh3>0.04]
    #   yh2[yh2>0.10] = ((yh1+yh2)/2)[yh2>0.10]

    detection = trigger_onset(yh1, args['detection_threshold'], args['detection_threshold'])
    pp_arr = _detect_peaks(yh2, mph=args['P_threshold'], mpd=1)
    ss_arr = _detect_peaks(yh3, mph=args['S_threshold'], mpd=1)

    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}
    if len(pp_arr) > 0:
        P_uncertainty = None

        for pick in range(len(pp_arr)):
            pauto = pp_arr[pick]

            if args['estimate_uncertainty'] and pauto:
                P_uncertainty = np.round(yh2_std[int(pauto)], 3)

            if pauto:
                P_prob = np.round(yh2[int(pauto)], 3)
                P_PICKS.update({pauto: [P_prob, P_uncertainty]})

    if len(ss_arr) > 0:
        S_uncertainty = None

        for pick in range(len(ss_arr)):
            sauto = ss_arr[pick]

            if args['estimate_uncertainty'] and sauto:
                S_uncertainty = np.round(yh3_std[int(sauto)], 3)

            if sauto:
                S_prob = np.round(yh3[int(sauto)], 3)
                S_PICKS.update({sauto: [S_prob, S_uncertainty]})

    if len(detection) > 0:
        D_uncertainty = None

        for ev in range(len(detection)):
            if args['estimate_uncertainty']:
                D_uncertainty = np.mean(yh1_std[detection[ev][0]:detection[ev][1]])
                D_uncertainty = np.round(D_uncertainty, 3)

            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)

            EVENTS.update({detection[ev][0]: [D_prob, D_uncertainty, detection[ev][1]]})

            # matching the detection and picks

    def pair_PS(l1, l2, dist):
        l1.sort()
        l2.sort()
        b = 0
        e = 0
        ans = []

        for a in l1:
            while l2[b] and b < len(l2) and a - l2[b] > dist:
                b += 1
            while l2[e] and e < len(l2) and l2[e] - a <= dist:
                e += 1
            ans.extend([[a, x] for x in l2[b:e]])

        best_pair = None
        for pr in ans:
            ds = pr[1] - pr[0]
            if abs(ds) < dist:
                best_pair = pr
                dist = ds
        return best_pair

    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]
        S_error = None
        P_error = None
        if int(ed - bg) >= 10:

            candidate_Ss = {}
            for Ss, S_val in S_PICKS.items():
                if Ss > bg and Ss < ed:
                    candidate_Ss.update({Ss: S_val})

            if len(candidate_Ss) > 1:
                # =============================================================================
                #                 Sr_st = 0
                #                 buffer = {}
                #                 for SsCan, S_valCan in candidate_Ss.items():
                #                     if S_valCan[0] > Sr_st:
                #                         buffer = {SsCan : S_valCan}
                #                         Sr_st = S_valCan[0]
                #                 candidate_Ss = buffer
                # =============================================================================
                candidate_Ss = {list(candidate_Ss.keys())[0]: candidate_Ss[list(candidate_Ss.keys())[0]]}

            if len(candidate_Ss) == 0:
                candidate_Ss = {None: [None, None]}

            candidate_Ps = {}
            for Ps, P_val in P_PICKS.items():
                if list(candidate_Ss)[0]:
                    if Ps > bg - 100 and Ps < list(candidate_Ss)[0] - 10:
                        candidate_Ps.update({Ps: P_val})
                else:
                    if Ps > bg - 100 and Ps < ed:
                        candidate_Ps.update({Ps: P_val})

            if len(candidate_Ps) > 1:
                Pr_st = 0
                buffer = {}
                for PsCan, P_valCan in candidate_Ps.items():
                    if P_valCan[0] > Pr_st:
                        buffer = {PsCan: P_valCan}
                        Pr_st = P_valCan[0]
                candidate_Ps = buffer

            if len(candidate_Ps) == 0:
                candidate_Ps = {None: [None, None]}

            # =============================================================================
            #             Ses =[]; Pes=[]
            #             if len(candidate_Ss) >= 1:
            #                 for SsCan, S_valCan in candidate_Ss.items():
            #                     Ses.append(SsCan)
            #
            #             if len(candidate_Ps) >= 1:
            #                 for PsCan, P_valCan in candidate_Ps.items():
            #                     Pes.append(PsCan)
            #
            #             if len(Ses) >=1 and len(Pes) >= 1:
            #                 PS = pair_PS(Pes, Ses, ed-bg)
            #                 if PS:
            #                     candidate_Ps = {PS[0] : candidate_Ps.get(PS[0])}
            #                     candidate_Ss = {PS[1] : candidate_Ss.get(PS[1])}
            # =============================================================================

            if list(candidate_Ss)[0] or list(candidate_Ps)[0]:
                matches.update({
                    bg: [ed,
                         EVENTS[ev][0],
                         EVENTS[ev][1],

                         list(candidate_Ps)[0],
                         candidate_Ps[list(candidate_Ps)[0]][0],
                         candidate_Ps[list(candidate_Ps)[0]][1],

                         list(candidate_Ss)[0],
                         candidate_Ss[list(candidate_Ss)[0]][0],
                         candidate_Ss[list(candidate_Ss)[0]][1],
                         ]})

                if sst and sst > bg and sst < EVENTS[ev][2]:
                    if list(candidate_Ss)[0]:
                        S_error = sst - list(candidate_Ss)[0]
                    else:
                        S_error = None

                if spt and spt > bg - 100 and spt < EVENTS[ev][2]:
                    if list(candidate_Ps)[0]:
                        P_error = spt - list(candidate_Ps)[0]
                    else:
                        P_error = None

                pick_errors.update({bg: [P_error, S_error]})

    return matches, pick_errors, yh3

class DataGeneratorTest(keras.utils.Sequence):
    """

    Keras generator with preprocessing. For testing.

    Parameters
    ----------
    list_IDsx: str
        List of trace names.

    file_name: str
        Path to the input hdf5 file.

    dim: tuple
        Dimension of input traces.

    batch_size: int, default=32
        Batch size.

    n_channels: int, default=3
        Number of channels.

    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.

    Returns
    --------
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.

    """

    def __init__(self,
                 list_IDs,
                 file_name,
                 dim,
                 batch_size=32,
                 n_channels=3,
                 norm_mode='max'):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_name = file_name
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def normalize(self, data, mode='max'):
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

    def __data_generation(self, list_IDs_temp):
        'readint the waveforms'

        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if ID.split('_')[-1] == 'EV':
                dataset = fl.get('data/' + str(ID))
                data = np.array(dataset)

            elif ID.split('_')[-1] == 'NO':
                dataset = fl.get('data/' + str(ID))
                data = np.array(dataset)

            if self.norm_mode:
                data = self.normalize(data, self.norm_mode)

            X[i, :, :] = data

        fl.close()

        return X

def tester(input_hdf5=None,
           input_testset=None,
           input_model=None,
           output_name=None,
           detection_threshold=0.20,
           P_threshold=0.1,
           S_threshold=0.1,
           number_of_plots=100,
           estimate_uncertainty=True,
           number_of_sampling=5,
           loss_weights=[0.05, 0.40, 0.55],
           loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
           input_dimention=(6000, 3),
           normalization_mode='std',
           mode='generator',
           batch_size=500,
           gpuid=None,
           gpu_limit=None):
    """

    Applies a trained model to a windowed waveform to perform both detection and picking at the same time.
    Parameters
    ----------
    input_hdf5: str, default=None
        Path to an hdf5 file containing only one class of "data" with NumPy arrays containing 3 component waveforms each 1 min long.
    input_testset: npy, default=None
        Path to a NumPy file (automaticaly generated by the trainer) containing a list of trace names.
    input_model: str, default=None
        Path to a trained model.

    output_dir: str, default=None
        Output directory that will be generated.

    output_probabilities: bool, default=False
        If True, it will output probabilities and estimated uncertainties for each trace into an HDF file.

    detection_threshold : float, default=0.3
        A value in which the detection probabilities above it will be considered as an event.

    P_threshold: float, default=0.1
        A value which the P probabilities above it will be considered as P arrival.
    S_threshold: float, default=0.1
        A value which the S probabilities above it will be considered as S arrival.

    number_of_plots: float, default=10
        The number of plots for detected events outputed for each station data.

    estimate_uncertainty: bool, default=False
        If True uncertainties in the output probabilities will be estimated.

    number_of_sampling: int, default=5
        Number of sampling for the uncertainty estimation.

    loss_weights: list, default=[0.03, 0.40, 0.58]
        Loss weights for detection, P picking, and S picking respectively.

    loss_types: list, default=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
        Loss types for detection, P picking, and S picking respectively.

    input_dimention: tuple, default=(6000, 3)
        Loss types for detection, P picking, and S picking respectively.
    normalization_mode: str, default='std'
        Mode of normalization for data preprocessing, 'max', maximum amplitude among three components, 'std', standard deviation.
    mode: str, default='generator'
        Mode of running. 'pre_load_generator' or 'generator'.

    batch_size: int, default=500
        Batch size. This wont affect the speed much but can affect the performance. A value beteen 200 to 1000 is recommanded.
    gpuid: int, default=None
        Id of GPU used for the prediction. If using CPU set to None.

    gpu_limit: int, default=None
        Set the maximum percentage of memory usage for the GPU.


    Returns
    --------
    ./output_name/X_test_results.csv: A table containing all the detection, and picking results. Duplicated events are already removed.

    ./output_name/X_report.txt: A summary of the parameters used for prediction and performance.

    ./output_name/figures: A folder containing plots detected events and picked arrival times.

    Notes
    --------
    Estimating the uncertainties requires multiple predictions and will increase the computational time.


    """

    args = {
        "input_hdf5": input_hdf5,
        "input_testset": input_testset,
        "input_model": input_model,
        "output_name": output_name,
        "detection_threshold": detection_threshold,
        "P_threshold": P_threshold,
        "S_threshold": S_threshold,
        "number_of_plots": number_of_plots,
        "estimate_uncertainty": estimate_uncertainty,
        "number_of_sampling": number_of_sampling,
        "loss_weights": loss_weights,
        "loss_types": loss_types,
        "input_dimention": input_dimention,
        "normalization_mode": normalization_mode,
        "mode": mode,
        "batch_size": batch_size,
        "gpuid": gpuid,
        "gpu_limit": gpu_limit
    }

    if args['gpuid']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args['gpuid'])
        tf.Session(config=tf.ConfigProto(log_device_placement=True))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit'])
        K.tensorflow_backend.set_session(tf.Session(config=config))

    save_dir = os.path.join(os.getcwd(), str(args['output_name']) + '_outputs')
    save_figs = os.path.join(save_dir, 'figures')

    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_figs)

    test = np.load(args['input_testset'])

    print('Loading the model ...', flush=True)
    model = load_model(args['input_model'], custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                                            'FeedForward': FeedForward,
                                                            'LayerNormalization': LayerNormalization,
                                                            'f1': f1
                                                            })

    model.compile(loss=args['loss_types'],
                  loss_weights=args['loss_weights'],
                  optimizer=Adam(lr=0.001),
                  metrics=[f1])

    print('Loading is complete!', flush=True)
    print('Testing ...', flush=True)
    print('Writting results into: " ' + str(args['output_name']) + '_outputs' + ' "', flush=True)

    start_training = time.time()

    csvTst = open(os.path.join(save_dir, 'X_test_results.csv'), 'w')
    test_writer = csv.writer(csvTst, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    test_writer.writerow(['network_code',
                          'ID',
                          'earthquake_distance_km',
                          'snr_db',
                          'trace_name',
                          'trace_category',
                          'trace_start_time',
                          'source_magnitude',
                          'p_arrival_sample',
                          'p_status',
                          'p_weight',
                          's_arrival_sample',
                          's_status',
                          's_weight',
                          'receiver_type',

                          'number_of_detections',
                          'detection_probability',
                          'detection_uncertainty',

                          'P_pick',
                          'P_probability',
                          'P_uncertainty',
                          'P_error',

                          'S_pick',
                          'S_probability',
                          'S_uncertainty',
                          'S_error'
                          ])
    csvTst.flush()

    plt_n = 0
    list_generator = generate_arrays_from_file(test, args['batch_size'])

    pbar_test = tqdm(total=int(np.ceil(len(test) / args['batch_size'])))
    for _ in range(int(np.ceil(len(test) / args['batch_size']))):
        pbar_test.update()
        new_list = next(list_generator)

        if args['mode'].lower() == 'pre_load_generator':
            params_test = {'dim': args['input_dimention'][0],
                           'batch_size': len(new_list),
                           'n_channels': args['input_dimention'][-1],
                           'norm_mode': args['normalization_mode']}
            test_set = {}
            fl = h5py.File(args['input_hdf5'], 'r')
            for ID in new_list:
                if ID.split('_')[-1] == 'EV':
                    dataset = fl.get('data/' + str(ID))
                elif ID.split('_')[-1] == 'NO':
                    dataset = fl.get('data/' + str(ID))
                test_set.update({str(ID): dataset})

            test_generator = PreLoadGeneratorTest(new_list, test_set, **params_test)

            if args['estimate_uncertainty']:
                pred_DD = []
                pred_PP = []
                pred_SS = []
                for mc in range(args['number_of_sampling']):
                    predD, predP, predS = model.predict_generator(test_generator)
                    pred_DD.append(predD)
                    pred_PP.append(predP)
                    pred_SS.append(predS)

                pred_DD = np.array(pred_DD).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_DD_mean = pred_DD.mean(axis=0)
                pred_DD_std = pred_DD.std(axis=0)

                pred_PP = np.array(pred_PP).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_PP_mean = pred_PP.mean(axis=0)
                pred_PP_std = pred_PP.std(axis=0)

                pred_SS = np.array(pred_SS).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_SS_mean = pred_SS.mean(axis=0)
                pred_SS_std = pred_SS.std(axis=0)

            else:
                pred_DD_mean, pred_PP_mean, pred_SS_mean = model.predict_generator(test_generator)
                pred_DD_mean = pred_DD_mean.reshape(pred_DD_mean.shape[0], pred_DD_mean.shape[1])
                pred_PP_mean = pred_PP_mean.reshape(pred_PP_mean.shape[0], pred_PP_mean.shape[1])
                pred_SS_mean = pred_SS_mean.reshape(pred_SS_mean.shape[0], pred_SS_mean.shape[1])

                pred_DD_std = np.zeros((pred_DD_mean.shape))
                pred_PP_std = np.zeros((pred_PP_mean.shape))
                pred_SS_std = np.zeros((pred_SS_mean.shape))

            for ts in range(pred_DD_mean.shape[0]):
                evi = new_list[ts]
                dataset = test_set[evi]

                try:
                    spt = int(dataset.attrs['p_arrival_sample']);
                except Exception:
                    spt = None

                try:
                    sst = int(dataset.attrs['s_arrival_sample']);
                except Exception:
                    sst = None

                matches, pick_errors, yh3 = picker(args, pred_DD_mean[ts], pred_PP_mean[ts], pred_SS_mean[ts],
                                                   pred_DD_std[ts], pred_PP_std[ts], pred_SS_std[ts], spt, sst)

                _output_writter_test(args, dataset, evi, test_writer, csvTst, matches, pick_errors)

                if plt_n < args['number_of_plots']:
                    _plotter(ts,
                             dataset,
                             evi,
                             args,
                             save_figs,
                             pred_DD_mean[ts],
                             pred_PP_mean[ts],
                             pred_SS_mean[ts],
                             pred_DD_std[ts],
                             pred_PP_std[ts],
                             pred_SS_std[ts],
                             matches)

                plt_n += 1


        else:
            params_test = {'file_name': str(args['input_hdf5']),
                           'dim': args['input_dimention'][0],
                           'batch_size': len(new_list),
                           'n_channels': args['input_dimention'][-1],
                           'norm_mode': args['normalization_mode']}

            test_generator = DataGeneratorTest(new_list, **params_test)

            if args['estimate_uncertainty']:
                pred_DD = []
                pred_PP = []
                pred_SS = []
                for mc in range(args['number_of_sampling']):
                    predD, predP, predS = model.predict_generator(generator=test_generator)
                    pred_DD.append(predD)
                    pred_PP.append(predP)
                    pred_SS.append(predS)

                pred_DD = np.array(pred_DD).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_DD_mean = pred_DD.mean(axis=0)
                pred_DD_std = pred_DD.std(axis=0)

                pred_PP = np.array(pred_PP).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_PP_mean = pred_PP.mean(axis=0)
                pred_PP_std = pred_PP.std(axis=0)

                pred_SS = np.array(pred_SS).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_SS_mean = pred_SS.mean(axis=0)
                pred_SS_std = pred_SS.std(axis=0)

            else:
                pred_DD_mean, pred_PP_mean, pred_SS_mean = model.predict_generator(generator=test_generator)
                pred_DD_mean = pred_DD_mean.reshape(pred_DD_mean.shape[0], pred_DD_mean.shape[1])
                pred_PP_mean = pred_PP_mean.reshape(pred_PP_mean.shape[0], pred_PP_mean.shape[1])
                pred_SS_mean = pred_SS_mean.reshape(pred_SS_mean.shape[0], pred_SS_mean.shape[1])

                pred_DD_std = np.zeros((pred_DD_mean.shape))
                pred_PP_std = np.zeros((pred_PP_mean.shape))
                pred_SS_std = np.zeros((pred_SS_mean.shape))

            test_set = {}
            fl = h5py.File(args['input_hdf5'], 'r')
            for ID in new_list:
                if ID.split('_')[-1] == 'EV':
                    dataset = fl.get('data/' + str(ID))
                elif ID.split('_')[-1] == 'NO':
                    dataset = fl.get('data/' + str(ID))
                test_set.update({str(ID): dataset})

            for ts in range(pred_DD_mean.shape[0]):
                evi = new_list[ts]
                dataset = test_set[evi]

                try:
                    spt = int(dataset.attrs['p_arrival_sample']);
                except Exception:
                    spt = None

                try:
                    sst = int(dataset.attrs['s_arrival_sample']);
                except Exception:
                    sst = None

                matches, pick_errors, yh3 = picker(args, pred_DD_mean[ts], pred_PP_mean[ts], pred_SS_mean[ts],
                                                   pred_DD_std[ts], pred_PP_std[ts], pred_SS_std[ts], spt, sst)

                _output_writter_test(args, dataset, evi, test_writer, csvTst, matches, pick_errors)

                if plt_n < args['number_of_plots']:
                    _plotter(dataset,
                             evi,
                             args,
                             save_figs,
                             pred_DD_mean[ts],
                             pred_PP_mean[ts],
                             pred_SS_mean[ts],
                             pred_DD_std[ts],
                             pred_PP_std[ts],
                             pred_SS_std[ts],
                             matches)

                plt_n += 1
    end_training = time.time()
    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta

    with open(os.path.join(save_dir, 'X_report.txt'), 'a') as the_file:
        the_file.write('================== Overal Info ==============================' + '\n')
        the_file.write('date of report: ' + str(datetime.datetime.now()) + '\n')
        the_file.write('input_hdf5: ' + str(args['input_hdf5']) + '\n')
        the_file.write('input_testset: ' + str(args['input_testset']) + '\n')
        the_file.write('input_model: ' + str(args['input_model']) + '\n')
        the_file.write('output_name: ' + str(args['output_name'] + '_outputs') + '\n')
        the_file.write('================== Testing Parameters =======================' + '\n')
        the_file.write('mode: ' + str(args['mode']) + '\n')
        the_file.write(
            'finished the test in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds, 2)))
        the_file.write('loss_types: ' + str(args['loss_types']) + '\n')
        the_file.write('loss_weights: ' + str(args['loss_weights']) + '\n')
        the_file.write('batch_size: ' + str(args['batch_size']) + '\n')
        the_file.write('total number of tests ' + str(len(test)) + '\n')
        the_file.write('gpuid: ' + str(args['gpuid']) + '\n')
        the_file.write('gpu_limit: ' + str(args['gpu_limit']) + '\n')
        the_file.write('================== Other Parameters =========================' + '\n')
        the_file.write('normalization_mode: ' + str(args['normalization_mode']) + '\n')
        the_file.write('estimate uncertainty: ' + str(args['estimate_uncertainty']) + '\n')
        the_file.write('number of Monte Carlo sampling: ' + str(args['number_of_sampling']) + '\n')
        the_file.write('detection_threshold: ' + str(args['detection_threshold']) + '\n')
        the_file.write('P_threshold: ' + str(args['P_threshold']) + '\n')
        the_file.write('S_threshold: ' + str(args['S_threshold']) + '\n')
        the_file.write('number_of_plots: ' + str(args['number_of_plots']) + '\n')


def _output_writter_test(args,
                         dataset,
                         evi,
                         output_writer,
                         csvfile,
                         matches,
                         pick_errors,
                         ):
    """

    Writes the detection & picking results into a CSV file.
    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.

    dataset: hdf5 obj
        Dataset object of the trace.
    evi: str
        Trace name.

    output_writer: obj
        For writing out the detection/picking results in the CSV file.

    csvfile: obj
        For writing out the detection/picking results in the CSV file.
    matches: dic
        Contains the information for the detected and picked event.

    pick_errors: dic
        Contains prediction errors for P and S picks.

    Returns
    --------
    X_test_results.csv


    """

    numberOFdetections = len(matches)

    if numberOFdetections != 0:
        D_prob = matches[list(matches)[0]][1]
        D_unc = matches[list(matches)[0]][2]

        P_arrival = matches[list(matches)[0]][3]
        P_prob = matches[list(matches)[0]][4]
        P_unc = matches[list(matches)[0]][5]
        P_error = pick_errors[list(matches)[0]][0]

        S_arrival = matches[list(matches)[0]][6]
        S_prob = matches[list(matches)[0]][7]
        S_unc = matches[list(matches)[0]][8]
        S_error = pick_errors[list(matches)[0]][1]

    else:
        D_prob = None
        D_unc = None

        P_arrival = None
        P_prob = None
        P_unc = None
        P_error = None

        S_arrival = None
        S_prob = None
        S_unc = None
        S_error = None

    if evi.split('_')[-1] == 'EV':
        network_code = dataset.attrs['network_code']
        source_id = dataset.attrs['source_id']
        source_distance_km = dataset.attrs['source_distance_km']
        snr_db = np.mean(dataset.attrs['snr_db'])
        trace_name = dataset.attrs['trace_name']
        trace_category = dataset.attrs['trace_category']
        trace_start_time = dataset.attrs['trace_start_time']
        source_magnitude = dataset.attrs['source_magnitude']
        p_arrival_sample = dataset.attrs['p_arrival_sample']
        p_status = dataset.attrs['p_status']
        p_weight = dataset.attrs['p_weight']
        s_arrival_sample = dataset.attrs['s_arrival_sample']
        s_status = dataset.attrs['s_status']
        s_weight = dataset.attrs['s_weight']
        receiver_type = dataset.attrs['receiver_type']

    elif evi.split('_')[-1] == 'NO':
        network_code = dataset.attrs['network_code']
        source_id = None
        source_distance_km = None
        snr_db = None
        trace_name = dataset.attrs['trace_name']
        trace_category = dataset.attrs['trace_category']
        trace_start_time = None
        source_magnitude = None
        p_arrival_sample = None
        p_status = None
        p_weight = None
        s_arrival_sample = None
        s_status = None
        s_weight = None
        receiver_type = dataset.attrs['receiver_type']

    if P_unc:
        P_unc = round(P_unc, 3)

    output_writer.writerow([network_code,
                            source_id,
                            source_distance_km,
                            snr_db,
                            trace_name,
                            trace_category,
                            trace_start_time,
                            source_magnitude,
                            p_arrival_sample,
                            p_status,
                            p_weight,
                            s_arrival_sample,
                            s_status,
                            s_weight,
                            receiver_type,

                            numberOFdetections,
                            D_prob,
                            D_unc,

                            P_arrival,
                            P_prob,
                            P_unc,
                            P_error,

                            S_arrival,
                            S_prob,
                            S_unc,
                            S_error,

                            ])

    csvfile.flush()


def _plotter(dataset, evi, args, save_figs, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, matches):
    """

    Generates plots.
    Parameters
    ----------
    dataset: obj
        The hdf5 obj containing a NumPy array of 3 component data and associated attributes.
    evi: str
        Trace name.
    args: dic
        A dictionary containing all of the input parameters.
    save_figs: str
        Path to the folder for saving the plots.
    yh1: 1D array
        Detection probabilities.
    yh2: 1D array
        P arrival probabilities.

    yh3: 1D array
        S arrival probabilities.
    yh1_std: 1D array
        Detection standard deviations.
    yh2_std: 1D array
        P arrival standard deviations.

    yh3_std: 1D array
        S arrival standard deviations.
    matches: dic
        Contains the information for the detected and picked event.


    """

    try:
        spt = int(dataset.attrs['p_arrival_sample']);
    except Exception:
        spt = None

    try:
        sst = int(dataset.attrs['s_arrival_sample']);
    except Exception:
        sst = None

    predicted_P = []
    predicted_S = []
    if len(matches) >= 1:
        for match, match_value in matches.items():
            if match_value[3]:
                predicted_P.append(match_value[3])
            else:
                predicted_P.append(None)

            if match_value[6]:
                predicted_S.append(match_value[6])
            else:
                predicted_S.append(None)

    data = np.array(dataset)

    fig = plt.figure()
    ax = fig.add_subplot(411)
    plt.plot(data[:, 0], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    legend_properties = {'weight': 'bold'}
    plt.title(str(evi))
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = None
    sl = None
    ppl = None
    ssl = None

    if dataset.attrs['trace_category'] == 'earthquake_local':
        if dataset.attrs['p_status'] == 'manual':
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_P_Arrival')
        else:
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Auto_P_Arrival')

        if dataset.attrs['s_status'] == 'manual':
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_S_Arrival')
        else:
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Auto_S_Arrival')
        if pl or sl:
            plt.legend(loc='upper right', borderaxespad=0., prop=legend_properties)

    ax = fig.add_subplot(412)
    plt.plot(data[:, 1], 'k')
    plt.tight_layout()
    if dataset.attrs['trace_category'] == 'earthquake_local':
        if dataset.attrs['p_status'] == 'manual':
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_P_Arrival')
        else:
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Auto_P_Arrival')

        if dataset.attrs['s_status'] == 'manual':
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_S_Arrival')
        else:
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Auto_S_Arrival')
        if pl or sl:
            plt.legend(loc='upper right', borderaxespad=0., prop=legend_properties)

    ax = fig.add_subplot(413)
    plt.plot(data[:, 2], 'k')
    plt.tight_layout()
    if len(predicted_P) > 0:
        ymin, ymax = ax.get_ylim()
        for pt in predicted_P:
            if pt:
                ppl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Predicted_P_Arrival')
    if len(predicted_S) > 0:
        for st in predicted_S:
            if st:
                ssl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Predicted_S_Arrival')

    if ppl or ssl:
        plt.legend(loc='upper right', borderaxespad=0., prop=legend_properties)

    ax = fig.add_subplot(414)
    x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)
    if args['estimate_uncertainty']:
        plt.plot(x, yh1, 'g--', alpha=0.5, linewidth=1.5, label='Detection')
        lowerD = yh1 - yh1_std
        upperD = yh1 + yh1_std
        plt.fill_between(x, lowerD, upperD, alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')

        plt.plot(x, yh2, 'b--', alpha=0.5, linewidth=1.5, label='P_probability')
        lowerP = yh2 - yh2_std
        upperP = yh2 + yh2_std
        plt.fill_between(x, lowerP, upperP, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

        plt.plot(x, yh3, 'r--', alpha=0.5, linewidth=1.5, label='S_probability')
        lowerS = yh3 - yh3_std
        upperS = yh3 + yh3_std
        plt.fill_between(x, lowerS, upperS, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.ylim((-0.1, 1.1))
        plt.tight_layout()
        plt.legend(loc='upper right', borderaxespad=0., prop=legend_properties)

    else:
        plt.plot(x, yh1, 'g--', alpha=0.5, linewidth=1.5, label='Detection')
        plt.plot(x, yh2, 'b--', alpha=0.5, linewidth=1.5, label='P_probability')
        plt.plot(x, yh3, 'r--', alpha=0.5, linewidth=1.5, label='S_probability')
        plt.tight_layout()
        plt.ylim((-0.1, 1.1))
        plt.legend(loc='upper right', borderaxespad=0., prop=legend_properties)

    fig.savefig(os.path.join(save_figs, str(evi.split('/')[-1]) + '.png'))