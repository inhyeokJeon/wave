from EQTransformer.core.tester import tester
import os
DIR = os.getcwd()
hdf5_path = os.path.join(DIR, 'ModelsAndSampleData/100samples.hdf5')

tester(input_hdf5='hdf5_path',
       input_testset='test_trainer_outputs/test.npy',
       input_model='test_trainer_outputs/models/test_trainer_001.h5',
       output_name='test_tester',
       detection_threshold=0.20,
       P_threshold=0.1,
       S_threshold=0.1,
       number_of_plots=3,
       estimate_uncertainty=True,
       number_of_sampling=2,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       batch_size=10,
       gpuid=None,
       gpu_limit=None)