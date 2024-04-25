import ssms
import pandas as pd
import numpy as np
from ssms.basic_simulators.simulator import simulator
from copy import deepcopy

#set up the generator and model configs
generator_config = deepcopy(ssms.config.data_generator_config['lan'])
generator_config['dgp_list'] = 'race_no_bias_5'
generator_config['output_folder'] = '/scratch/hyang336/working_dir/HSSM_dev/race_no_bias_5_dev/train_data/'
generator_config['n_parameter_sets'] = 10000
generator_config['n_samples'] = 1000

model_config = ssms.config.model_config['race_no_bias_5']

#get generator
my_generator = ssms.dataset_generators.lan_mlp.data_generator(generator_config = generator_config, model_config = model_config)
#generate data (there is some bug in the generating process, may have something to do with RT of 0. Since this package is being pushed few days, it is probably not worth it to fix it by myself)
No_error_cnt=0
while(True):  
  try:
    training_data = my_generator.generate_data_training_uniform(save=True)
    No_error_cnt+=1
  except Exception as e:
    print(f'Encountered error: {e}. Continuing to next iteration.')
  if No_error_cnt == 50:
    print('Reached maximum attempts without error')
    break
