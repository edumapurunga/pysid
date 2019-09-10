"""
    In this example we use the pysid library to estimate a SISO oe model
"""
#Import Libraries
from numpy.random import rand, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from pysid import oe                 #To estimate an arx model
#True System
#Number of inputs
nu = 1
#Number of outputs
ny = 1
#Orders
nf = 2       #This variable must be (ny x ny)
nb = 1       #This variable must be (ny x nu)
nk = 1       #This variable must be (ny x nu)
#with the following true parameters
Fo  = [1, -1.2, 0.36]
Bo = [0, 0.5, 0.1]
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.1]
#Generate the experiment
#The true system is generates by the following relation:
# S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 400
#Take u as uniform
u = -1 + 2*rand(N, nu)
#Generate gaussian white noise with standat deviation 0.1
e = 0.1*randn(N, ny)
#Calculate the y through S (OE: G(q) = B(q)/F(q) and H(q) = 1)
y = lfilter(Bo, Fo, u, axis=0) + e
#Estimate the model and get only the parameters
B, F = oe(nb, nf, nk, u, y)
