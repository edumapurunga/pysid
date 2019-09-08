"""
    In this example we use the SysID library to estimate an ARMA model
"""
#Import Libraries
from numpy.random import randn       #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from sysid import ma                 #To estimate an arx model
#True System
nb = 1
#with the following true parameters
Bo = [1, 0.55, 0.15] 
#True parameter vector
#The true system is generates by the following relation: 
# S: Ao(q)y(t) = Bo(q)e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 200
#Generate gaussian white noise with standat deviation 0.01
e = 0.01*randn(N, 1)
#Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
y = lfilter(Bo, [1], e, axis=0)
#Estimate the model using Durbin method
Bd = ma(nb, y)
#Estimate the model using Vocariance Method
Bvoc = ma(nb, y, 'vrm')
#Estimate the model using pem 
Bpem = ma(nb, y, 'pem')
