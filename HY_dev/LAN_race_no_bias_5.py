import ssms
import lanfactory 
import os
import numpy as np
from copy import deepcopy
import torch

# List of datafiles (here only one)
folder_ = '/scratch/hyang336/working_dir/HSSM_dev/race_no_bias_5_dev/data/'
file_list_ = [folder_ + file_ for file_ in os.listdir(folder_)]

# Training dataset
torch_training_dataset = lanfactory.trainers.torch_mlp.DatasetTorch(file_ids = file_list_,
                                                          batch_size = 128)

torch_training_dataloader = torch.utils.data.DataLoader(torch_training_dataset,
                                                         shuffle = True,
                                                         batch_size = None,
                                                         num_workers = 1,
                                                         pin_memory = True)

# Validation dataset
torch_validation_dataset = lanfactory.trainers.torch_mlp.DatasetTorch(file_ids = file_list_,
                                                          batch_size = 128)

torch_validation_dataloader = torch.utils.data.DataLoader(torch_validation_dataset,
                                                          shuffle = True,
                                                          batch_size = None,
                                                          num_workers = 1,
                                                          pin_memory = True)
                                                          
# SPECIFY NETWORK CONFIGS AND TRAINING CONFIGS

network_config = lanfactory.config.network_configs.network_config_mlp

#print('Network config: ')
#print(network_config)

train_config = lanfactory.config.network_configs.train_config_mlp

#print('Train config: ')
#print(train_config)

# LOAD NETWORK
net = lanfactory.trainers.TorchMLP(network_config = deepcopy(network_config),
                                   input_shape = torch_training_dataset.input_dim,
                                   save_folder = '/scratch/hyang336/working_dir/HSSM_dev/race_no_bias_5_dev/model/',
                                   generative_model_id = 'race_no_bias_5')

# SAVE CONFIGS
lanfactory.utils.save_configs(model_id = 'race_no_bias_5_torch_',
                              save_folder = '/scratch/hyang336/working_dir/HSSM_dev/race_no_bias_5_dev/config', 
                              network_config = network_config, 
                              train_config = train_config, 
                              allow_abs_path_folder_generation = True)
                              
# TRAIN MODEL 
model_trainer = lanfactory.trainers.ModelTrainerTorchMLP(
    model=net,
    train_config=train_config,
    train_dl=torch_training_dataloader,
    valid_dl=torch_validation_dataloader,
    allow_abs_path_folder_generation=False,
    pin_memory=True,
    seed=None)
    
model_trainer.train_and_evaluate(
    wandb_on=False,
    output_folder='/scratch/hyang336/working_dir/HSSM_dev/race_no_bias_5_dev/model/',
    output_file_id='race_no_bias_5')

