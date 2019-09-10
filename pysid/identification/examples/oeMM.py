"""
    In this example we use the pysid library to estimate a MIMO oe model
"""
#Import Libraries
from numpy import concatenate
from numpy.random import random, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from pysid import oe                 #To estimate an arx model
#True System
#Number of inputs
nu = 2
#Number of outputs
ny = 2
#Orders
nf = [[2, 2], [2, 2]]      #This variable must be (ny x nu)
nb = [[1, 1], [1, 1]]      #This variable must be (ny x nu)
nk = [[1, 1], [1, 1]]      #This variable must be (ny x nu)
#with the following true parameters
F11o = [1, -1.2, 0.36]
F12o = [1, -1.6, 0.84]
F21o = [1, -1.0, 0.25]
F22o = [1, -1.4, 0.49]
B11o = [0, 0.5, 0.1]
B12o = [0, 0.8,-0.4]
B21o = [0, 0.7, 0.2]
B22o = [0, 0.3,-0.2]
#True parameter vector
thetao = [-1.2, 0.36, -1.6, 0.84, -1, 0.25, -1.4, 0.49, 0.5, 0.1, 0.8, -0.4, 0.7, 0.2, 0.3, -0.2]
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
y1 = lfilter(B11o, F11o, u[:,0:1], axis=0) + lfilter(B12o, F12o, u[:,1:2], axis=0) + e[:,0:1]
y2 = lfilter(B21o, F21o, u[:,0:1], axis=0) + lfilter(B22o, F22o, u[:,1:2], axis=0) + e[:,1:2]
y = concatenate((y1, y2), axis=1)
#Estimate the model and get only the parameters
B, F = oe(nb, nf, nk, u, y)
