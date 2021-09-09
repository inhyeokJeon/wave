from EQTransformer.core.trainer import trainer
import os
DIR = os.getcwd()
hdf5_path = os.path.join(DIR, 'ModelsAndSampleData/100samples.hdf5')
csv_path = os.path.join(DIR, 'ModelsAndSampleData/100samples.csv')

trainer(input_hdf5=hdf5_path,
        input_csv=csv_path,
        output_name='test_trainer',
        cnn_blocks=2,
        lstm_blocks=1,
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