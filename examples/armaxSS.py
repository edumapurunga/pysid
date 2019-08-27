"""
    In this example we use the SysID library to estimate a SISO arx model
"""
#Import Libraries
from numpy import sqrt
from numpy.random import rand, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from sysid import armax                #To estimate an arx model
#True System
na = 2
nb = 1
nc = 2
nk = 1
#with the following true parameters
Ao = [1, -1.2, 0.36]
Bo = [0, 0.5, 0.1]
Co = [1., 0.8, -0.1]
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.1, 0.8, -0.1]
#Generate the experiment
#The true system is generates by the following relation: 
# S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 400
#Take u as uniform
u = -sqrt(3) + 2*sqrt(3)*rand(N, 1)
#Generate gaussian white noise with standat deviation 0.01
e = 0.01*randn(N, 1)
#Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
y = lfilter(Bo, Ao, u, axis=0) + lfilter(Co, Ao, e, axis=0)
#Estimate the model and get only the parameters
A, B, C = armax(na, nb, nc, nk, u, y)
