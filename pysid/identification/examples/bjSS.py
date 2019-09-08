"""
    In this example we use the SysID library to estimate a SISO BJ model
"""
#Import Libraries
from numpy.random import random, randn #To generate the experiment
from scipy.signal import lfilter       #To generate the data
from sysid import bj                   #To estimate an arx model
#True System
#Number of inputs
nu = 1
#Number of outputs
ny = 1
#Orders
nf = 2       #This variable must be (ny x ny)
nb = 1       #This variable must be (ny x nu)
nc = 2
nd = 2
nk = 1       #This variable must be (ny x nu)
#with the following true parameters
Fo  = [1, -1.2, 0.36]
Bo = [0, 0.5, 0.1]
Co = [1, 0.8, 0.2]
Do = [1, -1.6, 0.64]
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.1, 0.8, 0.2, -1.6, 0.64]
#Generate the experiment
#The true system is generates by the following relation: 
# S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 400
#Take u as uniform
u = -1 + 2*random((N, nu))
#Generate gaussian white noise with standat deviation 0.1
e = 0.1*randn(N, ny)
#Calculate the y through S (BJ: G(q) = B(q)/F(q) and H(q) = C(q)/D(q))
y = lfilter(Bo, Fo, u, axis=0) + lfilter(Co, Do, e, axis=0)
#Estimate the model and get only the parameters
B, C, D, F = bj(nb, nc, nd, nf, nk, u, y)
