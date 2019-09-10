"""
    In this example we use the pysid library to estimate a SISO Generbal Black Box model
"""
#Import Libraries
from numpy import convolve
from numpy.random import random, randn #To generate the experiment
from scipy.signal import lfilter       #To generate the data
from pysid import pem                   #To estimate an arx model
#True System
#Number of inputs
nu = 1
#Number of outputs
ny = 1
#Orders
na = 2       #This variable must be (ny x ny)
nb = 1       #This variable must be (ny x nu)
nc = 2
nd = 2
nf = 2
nk = 1       #This variable must be (ny x nu)
#with the following true parameters
Ao = [1, -1.2, 0.36]
Bo = [0,  0.5, 0.1]
Co = [1,  0.8, 0.2]
Do = [1, -1.6, 0.64]
Fo = [1, -1,   0.25]
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.1, 0.8, 0.2, -1.6, 0.64, -1, 0.25]
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
#Calculate the y through S (BJ: G(q) = B(q)/A(q)F(q) and H(q) = C(q)/A(q)D(q))
y = lfilter(Bo, convolve(Ao, Fo), u, axis=0) + lfilter(Co, convolve(Ao, Do), e, axis=0)
#Estimate the model and get only the parameters
thetahat = pem(na, nb, nc, nd, nf, nk, u, y)[0]
