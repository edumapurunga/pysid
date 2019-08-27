"""
    In this example we use the SysID library to estimate a SIMO oe model
"""
#Import Libraries
from numpy import concatenate
from numpy.random import random, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from sysid import oe                 #To estimate an arx model
#True System
#Number of inputs
nu = 1
#Number of outputs
ny = 2
#Orders
nf = [2, 2]       #This variable must be (ny x nu)
nb = [1, 1]       #This variable must be (ny x nu)
nk = [1, 1]       #This variable must be (ny x nu)
#with the following true parameters
F1o = [1, -1.2, 0.36]
F2o = [1, -1.4, 0.49]
B1o = [0, 0.5, 0.1]
B2o = [0, 0.3,-0.2]
#B2o = [0, 0.6,-0.2]
#True parameter vector
thetao = [-1.2, 0.36, -1.4, 0.49, 0.5, 0.1, 0.6, -0.2]
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
y1 = lfilter(B1o, F1o, u[:,0:1], axis=0) + e[:,0:1]
y2 = lfilter(B2o, F2o, u[:,0:1], axis=0) + e[:,1:2]
y = concatenate((y1, y2), axis=1)
#Estimate the model and get only the parameters
B, F = oe(nb, nf, nk, u, y)
