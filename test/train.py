import numpy as np
from tensorflow.keras.layers import Input
from tensorflow import keras
import os

from data_fetch import data_fetch
from datetime import datetime
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import cred2

def trainer(input_hdf5=None,
            input_csv=None,
            output_name=None,
            input_dimention=(6000, 3),
            cnn_blocks=5, # 5
            lstm_blocks=2, # 2
            padding='same',
            activation = 'relu',
            drop_rate=0.1,
            shuffle=True,
            label_type='gaussian',
            normalization_mode='std',
            augmentation=True,
            add_event_r=0.6,
            shift_event_r=0.99,
            add_noise_r=0.3,
            drop_channel_r=0.5,
            add_gap_r=0.2,
            coda_ratio=0.4,
            scale_amplitude_r=None,
            pre_emphasis=False,
            loss_weights = [0.05],
            # loss_weights=[0.05, 0.40, 0.55],
            loss_types=['binary_crossentropy'],
            # loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
            train_valid_test_split=[0.85, 0.05, 0.10],
            mode='generator',
            batch_size=200,
            epochs=200,
            monitor='val_loss',
            patience=12,
            gpuid=None,
            gpu_limit=None,
            use_multiprocessing=True):

    args = {
        "input_hdf5": input_hdf5,
        "input_csv": input_csv,
        "output_name": output_name,
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

    model = _build_model(args)

    logs = os.path.join(os.curdir, "my_logs",
                        "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logs, histogram_freq=1, profile_batch=10)

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_fetch('test_STEAD')
    
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb])

    save_dir = os.path.join(os.getcwd(), 'test_trainer' + '_outputs')

    np.save(save_dir + '/history', history)
    model.save(save_dir + '/final_model.h5')

def _build_model(args):
    inp = Input((6000,3), name='input')
    model = cred2(nb_filters=[8, 16, 16, 32, 32, 64, 64],
              kernel_size=[11, 9, 7, 7, 5, 5, 3],
              padding='same',
              activationf =args['activation'],
              cnn_blocks=args['cnn_blocks'],
              BiLSTM_blocks=args['lstm_blocks'],
              drop_rate=args['drop_rate'],
              loss_weights=args['loss_weights'],
              loss_types=args['loss_types'],
              kernel_regularizer=keras.regularizers.l2(1e-6),
              bias_regularizer=keras.regularizers.l1(1e-4)
               )(inp)

    model.summary()
    return model

trainer(input_hdf5='../ModelsAndSampleData/100samples.hdf5',
        input_csv='../ModelsAndSampleData/100samples.csv',
        output_name='test_trainer',
        cnn_blocks=2,
        lstm_blocks=3,
        padding='same',
        activation='relu',
        drop_rate=0.2,
        label_type='gaussian',
        add_event_r=0.6,
        add_gap_r=0.2,
        shift_event_r=0.9,
        add_noise_r=0.5,
        mode='generator',
        train_valid_test_split=[0.60, 0.20, 0.20],
        batch_size=20,
        epochs=10,
        patience=2,
        gpuid=None,
        gpu_limit=None)




