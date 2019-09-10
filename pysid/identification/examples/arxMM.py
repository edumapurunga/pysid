"""
    In this example we use the pysid library to estimate a MIMO arx model
"""
#Import Libraries
from numpy.random import rand, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from numpy import array, concatenate, zeros
from pysid import arx                #To estimate an arx model
#True System
#Number of inputs
nu = 2
#Number of outputs
ny = 2
#Orders
na = [[2, 2], [2, 2]]  #This variable must be (ny x ny)
nb = [[1, 1], [1, 1]]  #This variable must be (ny x nu)
nk = [[1, 1], [1, 1]]  #This variable must be (ny x nu)
#with the following true parameters
A11o = array(([-1.2, 0.36]))
A12o = array(([-0.2, 0.1]))
A21o = array(([-0.05, 0.09]))
A22o = array(([-1.4, 0.49]))
B11o = array(([0.5, 0.1]))
B12o = array(([1.0, 0.66]))
B21o = array(([0.8, 0.3]))
B22o = array(([0.65,0.2]))
#True parameter vector
thetao = [-1.2, 0.36, -0.2, 0.1, -0.05, 0.09, -1.4, 0.49, 0.5, 0.1, 1.0, 0.66, 0.8, 0.3, 0.65, 0.2]
#Generate the experiment
#The true system is generates by the following relation:
# S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 100
#Take u as uniform
u = -1 + 2*rand(N, nu)
#Generate gaussian white noise with standat deviation 0.01
e = 0.01*randn(N, ny)
#Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
y1 = zeros((N, 1))
y2 = zeros((N, 1))
#Simulate the true process
for i in range(2, N):
    y1[i] = -A11o.dot(y1[i-2:i][::-1]) - A12o.dot(y2[i-2:i][::-1]) + B11o.dot(u[i-2:i, 0][::-1]) + B12o.dot(u[i-2:i, 1][::-1]) + e[i, 0]
    y2[i] = -A21o.dot(y1[i-2:i][::-1]) - A22o.dot(y2[i-2:i][::-1]) + B21o.dot(u[i-2:i, 0][::-1]) + B22o.dot(u[i-2:i, 1][::-1]) + e[i, 1]
y = concatenate((y1, y2), axis=1)
#Estimate the model and get only the parameters
Ahat, Bhat = arx(na, nb, nk, u, y)
