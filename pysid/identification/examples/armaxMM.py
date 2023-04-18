"""
    In this example we use the pysid library to estimate a MIMO armax model
"""
#Import Libraries
from numpy import array, convolve, concatenate, zeros
from numpy.random import rand, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from pysid import armax                #To estimate an arx model
#True System
#Number of inputs
nu = 2
#Number of outputs
ny = 2
#Orders
na = [[2, 2], [2, 2]]       #This variable must be (ny x ny)
nb = [[1, 1], [1, 1]]       #This variable must be (ny x nu)
nk = [[1, 1], [1, 1]]       #This variable must be (ny x nu)
nc = [[2], [2]]                 #This variable must be (ny x 1)
#with the following true parameters
A1o  = array([1, -1.2, 0.36])
A12o = array([0, 0.09, -0.1])
A2o  = array([1, -1.6, 0.64])
A21o = array([0, 0.2, -0.01])
B11o = array([0, 0.5, 0.4])
B12o = array([0, 0.9, 0.8])
B21o = array([0, 0.2,-0.3])
B22o = array([0, 0.1,-0.8])
C1o  = array([1, 0.8,-0.1])
C2o  = array([1, 0.9,-0.2])
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.4, 0.2, -0.3, 0.8, -0.1]
# Generate the experiment
# The true system is generates by the following relation:
#    S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#       with u(t) the input and e white noise.
# Number of Samples
N = 1000
# Take u as uniform
u = -1 + 2*rand(N, nu)
# Generate gaussian white noise with standat deviation 0.01
e = 0.01*randn(N, ny)
# ARMAX A**-1*B u + A**-1*C
# Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
# Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))

det = convolve(A1o, A2o) - convolve(A12o, A21o) 
y1 = lfilter(convolve(A2o, B11o), det, u[:, 0:1], axis=0) + \
     lfilter(convolve(-A12o, B21o), det, u[:, 0:1], axis=0) + \
     lfilter(convolve(A2o, B12o), det, u[:, 1:2], axis=0) + \
     lfilter(convolve(-A12o, B22o), det, u[:, 1:2], axis=0) + \
     lfilter(convolve(A2o, C1o), det, e[:, 0:1], axis=0) + \
     lfilter(convolve(-A12o, C2o), det, e[:, 1:2], axis=0)
y2 = lfilter(convolve(-A21o, B11o), det, u[:, 0:1], axis=0) + \
     lfilter(convolve(A1o, B21o), det, u[:, 0:1], axis=0) + \
     lfilter(convolve(-A21o, B12o), det, u[:, 1:2], axis=0) + \
     lfilter(convolve(A1o, B22o), det, u[:, 1:2], axis=0) + \
     lfilter(convolve(-A21o, C1o), det, e[:, 0:1], axis=0) + \
     lfilter(convolve(A1o, C2o), det, e[:, 1:2], axis=0)

y = concatenate((y1, y2), axis=1)

#Estimate the model and get only the parameters
m = armax(na, nb, nc, nk, u, y)