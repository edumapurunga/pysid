"""
    In this example we use the pysid library to estimate a MISO bj model
"""
#Import Libraries
from numpy.random import random, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from pysid import bj                 #To estimate an arx model
#True System
#Number of inputs
nu = 2
#Number of outputs
ny = 1
#Orders
nf = [2, 2]       #This variable must be (ny x ny)
nb = [1, 1]       #This variable must be (ny x nu)
nc = 2
nd = 2
nk = [1, 1]       #This variable must be (ny x nu)
#with the following true parameters
F1o = [1, -1.2, 0.36]
F2o = [1, -1.4, 0.49]
B1o = [0, 0.5, 0.1]
B2o = [0, 0.3,-0.2]
Co = [1, 0.8, 0.16]
Do = [1, -1.8, 0.91]
#True parameter vector
thetao = [-1.2, 0.36, -1.4, 0.49, 0.5, 0.1, 0.3, -0.2, 0.8, 0.16, -1.8, 0.91]
#Generate the experiment
#The true system is generates by the following relation:
# S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 400
#Take u as uniform
u = -1 + 2*random((N, nu))
#u = ones((N, nu))
#Generate gaussian white noise with standard deviation 0.1
e = 0.01*randn(N, ny)
#Calculate the y through S (OE: G(q) = B(q)/F(q) and H(q) = 1)
y1 = lfilter(B1o, F1o, u[:,0:1], axis=0)
y2 = lfilter(B2o, F2o, u[:,1:2], axis=0)
y3 = lfilter(Co, Do, e[:,0:1], axis=0)
y_ = y1 + y2 + y3
y = lfilter(B1o, F1o, u[:,0:1], axis=0) + lfilter(B2o, F2o, u[:,1:2], axis=0) + lfilter(Co, Do, e[:,0:1], axis=0)
#Estimate the model and get only the parameters
m = bj(nb, nc, nd, nf, nk, u, y)