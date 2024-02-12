import ssms
import pandas as pd
import numpy as np
from ssms.basic_simulators.simulator import simulator
from copy import deepcopy

#set up the generator and model configs
generator_config = deepcopy(ssms.config.data_generator_config['lan'])
generator_config['dgp_list'] = 'race_no_bias_5'
generator_config['output_folder'] = '/scratch/hyang336/working_dir/HSSM_dev/race_no_bias_5_dev/'

model_config = ssms.config.model_config['race_no_bias_5']

#get generator
my_generator = ssms.dataset_generators.lan_mlp.data_generator(generator_config = generator_config, model_config = model_config)
#generate data
training_data = my_generator.generate_data_training_uniform(save=True)
