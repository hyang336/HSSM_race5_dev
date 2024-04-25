import lanfactory
import os
import numpy as np
from copy import deepcopy
import torch

model_data_folder = "/scratch/hyang336/working_dir/HSSM_dev/race_no_bias_5/"
model = "race_no_bias_5"
# MAKE DATALOADERS
# List of datafiles (here only one)
train_folder_ = model_data_folder + "train_data/" # + "/training_data_0_nbins_0_n_1000/"
train_file_list_ = [train_folder_ + file_ for file_ in os.listdir(train_folder_)]

valid_folder_ = model_data_folder + "validation_data/" # + "/training_data_0_nbins_0_n_1000/"
valid_file_list_ = [valid_folder_ + file_ for file_ in os.listdir(valid_folder_)]

# Training dataset
torch_training_dataset = lanfactory.trainers.DatasetTorch(
    file_ids=train_file_list_,
    batch_size=128,
    features_key="lan_data",
    label_key="lan_labels",
)

torch_training_dataloader = torch.utils.data.DataLoader(
    torch_training_dataset,
    shuffle=True,
    batch_size=None,
    num_workers=1,
    pin_memory=True,
)

# Validation dataset
torch_validation_dataset = lanfactory.trainers.DatasetTorch(
    file_ids=valid_file_list_,
    batch_size=128,
    features_key="lan_data",
    label_key="lan_labels",
)

torch_validation_dataloader = torch.utils.data.DataLoader(
    torch_validation_dataset,
    shuffle=True,
    batch_size=None,
    num_workers=1,
    pin_memory=True,
)

# SPECIFY NETWORK CONFIGS AND TRAINING CONFIGS

network_config = deepcopy(lanfactory.config.network_configs.network_config_mlp)
network_config["layer_sizes"] = [100, 100, 120, 1]
network_config["activations"] = ["tanh", "tanh", "tanh", "linear"]

print("Network config: ")
print(network_config)

train_config = deepcopy(lanfactory.config.network_configs.train_config_mlp)

print("Train config: ")
print(train_config)

# LOAD NETWORK
net = lanfactory.trainers.TorchMLP(
    network_config=deepcopy(network_config),
    input_shape=torch_training_dataset.input_dim,
    save_folder=model_data_folder + 'torch_model/',
    generative_model_id=model,
)

# SAVE CONFIGS
lanfactory.utils.save_configs(
    model_id=model + "_torch_",
    save_folder=model_data_folder + 'config/',
    network_config=network_config,
    train_config=train_config,
    allow_abs_path_folder_generation=True,
)

# TRAIN MODEL
model_trainer = lanfactory.trainers.ModelTrainerTorchMLP(
    model=net,
    train_config=train_config,
    train_dl=torch_training_dataloader,
    valid_dl=torch_validation_dataloader,
    allow_abs_path_folder_generation=True,
    pin_memory=True,
    seed=None,
)

# model_trainer.train_model(save_history=True, save_model=True, verbose=0)
model_trainer.train_and_evaluate(
    wandb_on=False,
    output_folder=model_data_folder + 'torch_model/',
    output_file_id=model,
)

