#generate data using race_4 model, there are separate v and z for each accumulator, but a and t are shared
#!simple version, just linear relationship between (independent) v's and covariate
from ssms.basic_simulators import simulator
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.special import beta
import hssm
import bambi as bmb
import arviz as az
from matplotlib import pyplot as plt
import pymc as pm

outdir='/scratch/hyang336/working_dir/HSSM_dev/race_4_LAN/'

#------------Not sure if the failed param recovery was due to anticorrelated IVs or the particular nonlinear model
#------------So we try both
# 20231020: based on https://github.com/lnccbrown/HSSM/blob/afa2abac8b6c3a5f19d292337096d7c372f4f49e/docs/api/defaults.md?plain=1#L123
# There seems to be a bound on the parameters, trying to keep the parameters in bound
a0=0.4
b0=1
#intercept0=np.log(1/beta(a0,b0))
a1=1
b1=1
#intercept1=np.log(1/beta(a1,b1))
a2=0.5
b2=1.5
#intercept2=np.log(1/beta(a2,b2))
a3=1.5
b3=0.4
#intercept3=np.log(1/beta(a3,b3))
simneural = np.random.uniform(0, 1, size=1000)#some fake neural data
simneural2= np.random.uniform(0, 1, size=1000)

# #simulate simple linear relationship with 2 colinear IVs
# v0=intercept0 + (a0-1)*simneural + (b0-1)*(1-simneural)
# v1=intercept1 + (a1-1)*simneural + (b1-1)*(1-simneural)
# v2=intercept2 + (a2-1)*simneural + (b2-1)*(1-simneural)
# v3=intercept3 + (a3-1)*simneural + (b3-1)*(1-simneural)

#simulate simple linear relationship with 2 independent IVs
v0=0.5 + a0*simneural + b0*simneural2
v1=0.2 + a1*simneural + b1*simneural2
v2=0.3 + a2*simneural + b2*simneural2
v3=0.4 + a3*simneural + b3*simneural2

#generate trial-wise parameters with fixed a, z, and t, and bnoundary_param (in that order), assumed to take the form theta in radian
true_values = np.column_stack(
    [v0,v1,v2,v3, np.repeat([[1.5, 0.0, 0.5,0.1]], axis=0, repeats=1000)]
)


# Get mode simulations
race4_v = simulator.simulator(true_values, model="race_no_bias_angle_4", n_samples=1)

dataset_race4_v = pd.DataFrame(
    {
        "rt": race4_v["rts"].flatten(),
        "response": race4_v["choices"].flatten(),
        "x": simneural,
        "y": simneural2
    }
)


#estimate parameters based on data
model_race4_v = hssm.HSSM(
    data=dataset_race4_v,
    model='race_no_bias_angle_4',
    a=1.5,
    include=[
        {
            "name": "v0",
            "prior":{"name": "Uniform", "lower": -5, "upper": 5},
            "formula": "v0 ~ 1 + x + y",
            "link": "identity",
        },
        {
            "name": "v1",
            "prior":{"name": "Uniform", "lower": -5, "upper": 5},
            "formula": "v1 ~ 1 + x + y",
            "link": "identity",
        },
        {
            "name": "v2",
            "prior":{"name": "Uniform", "lower": -5, "upper": 5},
            "formula": "v2 ~ 1 + x + y",
            "link": "identity",
        },
        {
            "name": "v3",
            "prior":{"name": "Uniform", "lower": -5, "upper": 5},
            "formula": "v3 ~ 1 + x + y",
            "link": "identity",
        }
    ],
)

#model graph


#sample from the model, 2000-5000 is not enough for the chain the converge
infer_data_race4_v = model_race4_v.sample(
    step=pm.Slice(model=model_race4_v.pymc_model), sampler="mcmc", chains=2, cores=None, draws=5000, tune=10000
)

#save model
az.to_netcdf(infer_data_race4_v,outdir+'sample5000_10000_trace_simpleIND_1000data_inbound_Fixed_a_SliceSampler.nc4')

#load model
#infer_data_race4_v=az.from_netcdf('/home/hyang336/HSSM_race5_dev/HY_dev/race_4_LAN/sample50_trace.nc4')

#diagnostic plots
az.plot_trace(
    infer_data_race4_v,
    var_names="~log_likelihood",  # we exclude the log_likelihood traces here
)
plt.savefig(outdir+'posterior_diagnostic5000_10000_simpleIND_1000data_inbound_Fixed_a_SliceSampler.png')

#fit summary
res_sum=az.summary(model_race4_v.traces)
res_sum.to_csv(outdir+'summary5000_10000_simpleIND_1000data_inbound_Fixed_a_SliceSampler.csv')
#res_slope=res_sum[res_sum.iloc[:,0].str.contains("_x|_y")]
#res_sum.loc[['v0_x','v0_y','v1_x','v1_y','v2_x','v2_y','v3_x','v3_y']]


#parameter recovery is pretty bad...

