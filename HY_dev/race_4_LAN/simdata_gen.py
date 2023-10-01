#generate data using race_4 model, there are separate v and z for each accumulator, but a and t are shared
from ssms.basic_simulators import simulator
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.special import beta
import hssm
import bambi as bmb
import arviz as az
from matplotlib import pyplot as plt

outdir='/home/hyang336/HSSM_race5_dev/HY_dev/race_4_LAN/'
#--------------------------------------We can try several generative model--------------------------------###
#fake trialwise neural data, the 4 accumulators are simulated to have monotonic or nonmonotonic relationships with 
#(log-transformed) neural data. This is controlled by take the beta distribution and log transform it, making it a 
#simple linear regression on the log-transformed neural data. The intercept becomes the log transform of 1 over beta 
#function, evaluated at the parameters for the beta distribution (a and b). This intercept was the normalizing factor
#before the log transformation to make sure the function was a distibution (i.e. integrate to 1). Note, larger values
#of a or b will result in large positive or negative value of v down the line, may need to find a way to rescale it...
a0=0.4
b0=1
intercept0=np.log(1/beta(a0,b0))
a1=1.5
b1=3
intercept1=np.log(1/beta(a1,b1))
a2=3
b2=1.5
intercept2=np.log(1/beta(a2,b2))
a3=1
b3=0.4
intercept3=np.log(1/beta(a3,b3))
simneural = np.random.uniform(0, 1, size=1000)

#simulate linear relationship between v and log-transformed neural data, following HSSM tutorial (i.e. no added noise at this step)
#now we need a log link function since we model log(v)=a+b*log(x)+c*log(1-x)
v0=np.exp(intercept0 + (a0-1)*np.log(simneural) + (b0-1)*np.log(1-simneural))
v1=np.exp(intercept1 + (a1-1)*np.log(simneural) + (b1-1)*np.log(1-simneural))
v2=np.exp(intercept2 + (a2-1)*np.log(simneural) + (b2-1)*np.log(1-simneural))
v3=np.exp(intercept3 + (a3-1)*np.log(simneural) + (b3-1)*np.log(1-simneural))

#generate trial-wise parameters with fixed a, z, and t, and bnoundary_param, assumed to take the form theta in radian
true_values = np.column_stack(
    [v0,v1,v2,v3, np.repeat([[1.5, 0.0, 0.5,0.1]], axis=0, repeats=1000)]
)


# Get mode simulations
race4_v = simulator.simulator(true_values, model="race_no_bias_angle_4", n_samples=1)

dataset_race4_v = pd.DataFrame(
    {
        "rt": race4_v["rts"].flatten(),
        "response": race4_v["choices"].flatten(),
        "x": np.log(simneural),
        "y": np.log(1-simneural)
    }
)


#estimate parameters based on data
model_race4_v = hssm.HSSM(
    data=dataset_race4_v,
    model='race_no_bias_angle_4',
    include=[
        {
            "name": "v0",
            "prior":{"name": "Uniform", "lower": -12, "upper": 12},
            "formula": "v0 ~ 1 + x + y",
            "link": "log",
        },
        {
            "name": "v1",
            "prior":{"name": "Uniform", "lower": -12, "upper": 12},
            "formula": "v1 ~ 1 + x + y",
            "link": "log",
        },
        {
            "name": "v2",
            "prior":{"name": "Uniform", "lower": -12, "upper": 12},
            "formula": "v2 ~ 1 + x + y",
            "link": "log",
        },
        {
            "name": "v3",
            "prior":{"name": "Uniform", "lower": -12, "upper": 12},
            "formula": "v3 ~ 1 + x + y",
            "link": "log",
        }
    ],
)

#model graph


#sample from the model, 500-500 is not enough for the chain the converge
infer_data_race4_v = model_race4_v.sample(
    sampler="nuts_numpyro", chains=1, cores=1, draws=2500, tune=2500
)

#save model
az.to_netcdf(infer_data_race4_v,outdir+'sample2500_trace.nc4')

#load model
#infer_data_race4_v=az.from_netcdf('/home/hyang336/HSSM_race5_dev/HY_dev/race_4_LAN/sample50_trace.nc4')

#diagnostic plots
az.plot_trace(
    infer_data_race4_v,
    var_names="~log_likelihood",  # we exclude the log_likelihood traces here
)
plt.savefig(outdir+'posterior_diagnostic.png')

#fit summary
res_sum=az.summary(model_race4_v.traces)
res_sum.to_csv(outdir+'summary.csv')
#res_slope=res_sum[res_sum.iloc[:,0].str.contains("_x|_y")]
#res_sum.loc[['v0_x','v0_y','v1_x','v1_y','v2_x','v2_y','v3_x','v3_y']]


#parameter recovery is pretty bad...

