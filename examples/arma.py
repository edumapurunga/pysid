"""
    In this example we use the SysID library to estimate an ARMA model
"""
#Import Libraries
from numpy.random import randn       #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from sysid import arma           #To estimate an arx model
#True System
na = 2
nb = 1
#with the following true parameters
Ao = [1, -1.2, 0.6]
Bo = [1, 0.5, 0.8] 
#True parameter vector
#The true system is generates by the following relation: 
# S: Ao(q)y(t) = Bo(q)e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 100
#Generate gaussian white noise with standat deviation 0.01
e = 0.01*randn(N, 1)
#Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
y = lfilter(Bo, Ao, e, axis=0)
#Estimate the model using Yule-Walker method
Ayw, Byw = arma(na, nb, y)
#Estimate the model using Burg method
#Aburg = burg(y, na)
#Estimate the model using pem 
#Apem = ar(na, y, 'pem')
