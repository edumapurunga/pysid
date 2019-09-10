"""
    In this example we use the pysid library to estimate a MISO armax model
"""
#Import Libraries
from numpy.random import rand, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from pysid import armax                #To estimate an arx model
#True System
#Number of inputs
nu = 2
#Number of outputs
ny = 1
#Orders
na = 2       #This variable must be (ny x ny)
nb = [1, 1]  #This variable must be (ny x nu)
nk = [1, 1]  #This variable must be (ny x nu)
nc = 2
#with the following true parameters
Ao  = [1, -1.2, 0.36]
B0o = [0, 0.5, 0.4]
B1o = [0, 0.2,-0.3]
Co  = [1, 0.8,-0.1]
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.4, 0.2, -0.3, 0.8, -0.1]
#Generate the experiment
#The true system is generates by the following relation:
# S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 200
#Take u as uniform
u = -1 + 2*rand(N, nu)
#Generate gaussian white noise with standat deviation 0.01
e = 0.01*randn(N, ny)
#Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
y = lfilter(B0o, Ao, u[:,0:1], axis=0) + lfilter(B1o, Ao, u[:,1:2], axis=0) + lfilter(Co, Ao, e[:,0:1], axis=0)
#Estimate the model and get only the parameters
A, B, C = armax(na, nb, nc, nk, u, y)
