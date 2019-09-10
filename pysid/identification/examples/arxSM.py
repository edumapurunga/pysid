"""
    In this example we use the pysid library to estimate a SIMO arx model
"""
#Import Libraries
from numpy import zeros, concatenate, dot
from numpy.random import rand, randn #To generate the experiment
from pysid import arx                #To estimate an arx model
#True System
#Number of inputs
nu = 1
#Number of outputs
ny = 2
#Orders
na = [[2, 2], [2, 2]]      #This variable must be (ny x ny)
nb = [[1], [1]]             #This variable must be (ny x nu)
nk = [[1], [1]]             #This variable must be (ny x nu)
#with the following true parameters
A11o = [1, -1.2, 0.36]
A12o = [0, 0.04,-0.05]
A21o = [0, 0.09, 0.03]
A22o = [1, -1.6, 0.64]
B1o = [0, 1.0, 0.5]
B2o = [0, 0.8, 0.3]
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.1, 1, 0.5, 0.8, 0.3]
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
y1 = zeros((N, 1))
y2 = zeros((N, 1))
#Simulate the true process
for i in range(2, N):
    y1[i] = dot(A11o[1:], -y1[i-2:i][::-1]) + dot(A12o[1:], -y2[i-2:i][::-1]) + dot(B1o[1:], u[i-2:i, 0][::-1])  + e[i, 0]
    y2[i] = dot(A21o[1:], -y1[i-2:i][::-1]) + dot(A22o[1:], -y2[i-2:i][::-1]) + dot(B2o[1:], u[i-2:i, 0][::-1])  + e[i, 1]
y = concatenate((y1, y2), axis=1)
#Estimate the model and get only the parameters
A, B = arx(na, nb, nk, u, y)
