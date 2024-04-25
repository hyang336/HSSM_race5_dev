import ssms
import lanfactory
import os
import numpy as np
from copy import deepcopy
import torch

# MAKE CONFIGS
model = "race_no_bias_4"
# Initialize the generator config (for MLP LANs)
generator_config = deepcopy(ssms.config.data_generator_config["lan"])
# Specify generative model (one from the list of included models mentioned above)
generator_config["model"] = model
# Specify number of parameter sets to simulate
#generator_config["n_parameter_sets"] = 1500000
# Specify how many samples a simulation run should entail
#generator_config["n_samples"] = 100000
# Specify folder in which to save generated data
generator_config["output_folder"] = "/scratch/hyang336/working_dir/HSSM_dev/race_4_LAN/" + model + "/"

# Make model config dict
model_config = ssms.config.model_config[model]


# MAKE DATALOADERS

# List of datafiles (here only one)
folder_ = generator_config["output_folder"] + "train_data/" # + "/training_data_0_nbins_0_n_1000/"
file_list_ = [folder_ + file_ for file_ in os.listdir(folder_)]

# Training dataset
torch_training_dataset = lanfactory.trainers.DatasetTorch(
    file_ids=file_list_,
    batch_size=1024,
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
    file_ids=file_list_,
    batch_size=1024,
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
network_config["layer_sizes"] = [100, 100, 100, 1]
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
    save_folder=generator_config["output_folder"] + 'torch_model/',
    generative_model_id=model,
)

# SAVE CONFIGS
lanfactory.utils.save_configs(
    model_id=model + "_torch_",
    save_folder=generator_config["output_folder"] + 'config/',
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
    output_folder=generator_config["output_folder"] + 'torch_model/',
    output_file_id=model,
)

