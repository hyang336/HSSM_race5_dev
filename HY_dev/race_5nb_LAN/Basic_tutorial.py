import ssms
import os
import numpy as np
from copy import deepcopy
import torch

# MAKE CONFIGS
model = "race_no_bias_5"
# Initialize the generator config (for MLP LANs)
generator_config = deepcopy(ssms.config.data_generator_config["lan"])
# Specify generative model (one from the list of included models mentioned above)
generator_config["model"] = model
# Specify number of parameter sets to simulate (1.5 million is what they used in the paper)
generator_config["n_parameter_sets"] = 5000
# Specify how many samples per parameter set to simulate
generator_config["n_training_samples_by_parameter_set"] = 100000
# Specify folder in which to save generated data
generator_config["output_folder"] = "/scratch/hyang336/working_dir/HSSM_dev/" + model + "/train_data/"

# Make model config dict
model_config = ssms.config.model_config[model]

# MAKE DATA

my_dataset_generator = ssms.dataset_generators.lan_mlp.data_generator(
    generator_config=generator_config, model_config=model_config
)

# need to run 1.5 million/n_parameter_sets times of this
training_data = my_dataset_generator.generate_data_training_uniform(save=True)
    

