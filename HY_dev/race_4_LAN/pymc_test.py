# testing if the model based on beta PDF is estimatable by PyMC
import pymc as pm
import numpy as np

#######specification of true parameters######
a0=0.1
b0=1
#intercept0=np.log(1/beta(a0,b0))
intercept0=1

a1=1.5
b1=3
#intercept1=np.log(1/beta(a1,b1))
intercept1=3.5

a2=3
b2=1.5
#intercept2=np.log(1/beta(a2,b2))
intercept2=3.5

a3=1
b3=0.1
#intercept3=np.log(1/beta(a3,b3))
intercept3=1

simneural = np.random.uniform(0, 1, size=2000)

noise=np.random.default_rng(8927)
sigma = 1

#simulate linear relationship between v and log-transformed neural data, following HSSM tutorial (i.e. no added noise at this step)
#now we need a log link function since we model log(v)=a+b*log(x)+c*log(1-x)
v0=np.exp(intercept0 + a0*np.log(simneural) + b0*np.log(1-simneural)) + noise.normal(size=2000) * sigma
v1=np.exp(intercept1 + a1*np.log(simneural) + b1*np.log(1-simneural)) + noise.normal(size=2000) * sigma
v2=np.exp(intercept2 + a2*np.log(simneural) + b2*np.log(1-simneural)) + noise.normal(size=2000) * sigma
v3=np.exp(intercept3 + a3*np.log(simneural) + b3*np.log(1-simneural)) + noise.normal(size=2000) * sigma


####define pymc model#####
model=pm.Model()

with model:
	#priors
	inter=pm.Normal("intercept",mu=0,sigma=10)
	slopes=pm.Normal("slopes",mu=0,sigma=10,shape=2)
	sigma = pm.HalfNormal("sigma",sigma=1)

	#expected value
	mu = inter + slopes[0] * np.log(simneural) + slopes[1] * np.log(1-simneural)

	#likelihood (this is the difficult part, since the likelihood is from  LAN)
