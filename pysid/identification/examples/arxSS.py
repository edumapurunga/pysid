"""
    In this example we use the pysid library to estimate a SISO arx model
"""
#Import Libraries
from numpy.random import rand, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from pysid import arx                #To estimate an arx model
#True System
na = 2
nb = 1
nk = 1
#with the following true parameters
Ao = [1, -1.2, 0.36]
Bo = [0, 0.5, 0.1]
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.1]
#Generate the experiment
#The true system is generates by the following relation:
# S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 100
#Take u as uniform
u = -1 + 2*rand(N, 1)
#Generate gaussian white noise with standat deviation 0.01
e = 0.01*randn(N, 1)
#Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
y = lfilter(Bo, Ao, u, axis=0) + lfilter([1], Ao, e, axis=0)
#Estimate the model and get only the parameters
m = arx(na, nb, nk, u, y)
