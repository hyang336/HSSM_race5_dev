import ssms
import lanfactory
import os
import numpy as np
from copy import deepcopy
import torch
import pandas as pd
from ssms.basic_simulators.simulator import simulator
import matplotlib.pyplot as plt

model = "race_no_bias_4"

# Load trained network
network_path_list = os.listdir("/scratch/hyang336/working_dir/HSSM_dev/race_4_LAN/race_no_bias_4/torch_model/")
network_file_path = [
    "/scratch/hyang336/working_dir/HSSM_dev/race_4_LAN/race_no_bias_4/torch_model/" + file_
    for file_ in network_path_list
    if "state_dict" in file_
][0]

network_config_list = os.listdir("/scratch/hyang336/working_dir/HSSM_dev/race_4_LAN/race_no_bias_4/config/")
network_config_path = [
    "/scratch/hyang336/working_dir/HSSM_dev/race_4_LAN/race_no_bias_4/config/" + file_
    for file_ in network_config_list
    if "network_config" in file_
][0]

network_config=pd.read_pickle(network_config_path)

network = lanfactory.trainers.LoadTorchMLPInfer(
    model_file_path=network_file_path,
    network_config=network_config,
    input_dim=9, #hard-coded for now since I want to separate infer from training 
)

#generate test data
data = pd.DataFrame(
    np.zeros((2000, 9), dtype=np.float32),
    columns=["v0", "v1", "v2", "v3", "a", "z", "t", "rt", "choice"],
)
data["v0"] = 0.1
data["v0"] = 1.0
data["v0"] = 1.5
data["v0"] = 0.4
data["a"] = 1.25
data["z"] = 0.5
data["t"] = 0.2
data["rt"].iloc[:500] = np.linspace(0, 5, 500) #these are just for x-axes
data["rt"].iloc[500:1000] = np.linspace(0, 5, 500)
data["rt"].iloc[1000:1500] = np.linspace(0, 5, 500)
data["rt"].iloc[1500:2000] = np.linspace(0, 5, 500)
data["choice"].iloc[:500] = 0
data["choice"].iloc[500:1000] = 1
data["choice"].iloc[1000:1500] = 2
data["choice"].iloc[1500:2000] = 3

# Network predictions
predict_on_batch_out = network.predict_on_batch(data.values.astype(np.float32))

# generative model
# sim_out = {}
# for i in range(10):
#   print(i)
#   sim_out[i] = simulator(model=model, theta=data.values[0, :-2], n_samples=2000)


# Need to save the figure to disk since you cannot directly visualize on Graham


### Plot network predictions
# Get unique choices
unique_choices = data["choice"].unique()

# Create subplots
fig, axs = plt.subplots(len(unique_choices))

# Loop over unique choices
for i, choice in enumerate(unique_choices):
    # Filter data for current choice
    filtered_data = data[data["choice"] == choice]
    
    # Plot data on separate subplot
    axs[i].plot(
        filtered_data["rt"],
        np.exp(predict_on_batch_out[filtered_data.index]),
        color="black",
        label=f"network - choice {choice}",
    )
    axs[i].legend()

        # Loop over simulations
    for j in range(10):
        my_seed = np.random.choice(1000000)
        sim_out = simulator(
            model=model, theta=data.values[0, :-2], n_samples=2000, random_state=my_seed
        )

        # Filter simulated data for current choice
        sim_filtered = sim_out["rts"][sim_out["choices"] == choice]

        # Plot simulated data on current subplot
        axs[i].hist(
            sim_filtered,
            bins=100,
            histtype="step",
            label="simulations",
            color="blue",
            alpha=0.2,
            density=True,
        )

plt.legend()
plt.xlabel("rt")
plt.ylabel("likelihod")

plt.savefig('/scratch/hyang336/working_dir/HSSM_dev/race_4_LAN/race_no_bias_4/race_4_tutorial.png')#this doesn't look so good...
