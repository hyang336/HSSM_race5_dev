#generate data using race_no_bias_angle_4 model
from ssms.basic_simulators import simulator
import numpy as np
import pandas as pd
# a_true_sample=np.random.normal(loc=2,scale=0.3,size=1000)
# v_true_sample=np.random.normal(loc=2,scale=0.3,size=1000)
# z_true_sample=np.random.normal(loc=2,scale=0.3,size=1000)
# t_true_sample=np.random.normal(loc=2,scale=0.3,size=1000)

# par_mat = np.zeros((1000, 4))
# par_mat[:, 0] = v_true_sample
# par_mat[:, 1] = a_true_sample
# par_mat[:, 2] = z_true_sample
# par_mat[:, 3] = t_true_sample

# # Simulate data
# sim_out_trialwise = simulator.simulator(
#     theta=par_mat,  # parameter_matrix
#     model="ddm",  # specify model (many are included in ssms)
#     n_samples=1,  # number of samples for each set of parameters
# )