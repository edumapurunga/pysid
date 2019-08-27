"""
    In this example we use the SysID library to estimate a SIMO armax model
"""
#Import Libraries
from numpy import concatenate, dot, zeros, sqrt
from numpy.random import rand, randn #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from sysid import armax                #To estimate an arx model
#True System
#Number of inputs
nu = 1
#Number of outputs
ny = 2
#Orders
na = [[2, 2], [2, 2]]       #This variable must be (ny x ny)
nb = [1, 1]                 #This variable must be (ny x nu)
nk = [1, 1]                 #This variable must be (ny x nu)
nc = [2, 2]                 #This variable must be (ny x 1)
#with the following true parameters
A1o  = [1, -1.2, 0.36]
A12o = [0, 0.09, -0.1]
A2o  = [1, -1.6, 0.64]
A21o = [0, 0.2, -0.01]
B1o = [0, 0.5, 0.4]
B2o = [0, 0.2,-0.3]
C1o  = [1, 0.8,-0.1]
C2o  = [1, 0.9,-0.2]
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.4, 0.2, -0.3, 0.8, -0.1]
#Generate the experiment
#The true system is generates by the following relation: 
# S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 400
#Take u as uniform
u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)
#Generate gaussian white noise with standat deviation 0.01
e = 0.01*randn(N, ny)
#Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
#Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
y1 = zeros((N, 1))
y2 = zeros((N, 1))
v1 = lfilter(C1o, [1], e[:,0:1], axis=0)
v2 = lfilter(C2o, [1], e[:,1:2], axis=0)
#Simulate the true process
for i in range(2, N):
    y1[i] = -dot(A1o[1:3] ,y1[i-2:i][::-1]) - dot(A12o[1:3],y2[i-2:i][::-1]) + dot(B1o[1:3], u[i-2:i, 0][::-1])
    y2[i] = -dot(A21o[1:3], y1[i-2:i][::-1]) - dot(A2o[1:3], y2[i-2:i][::-1]) + dot(B2o[1:3], u[i-2:i, 0][::-1]) 
y = concatenate((y1+v1, y2+v2), axis=1)
#Estimate the model and get only the parameters
A, B, C = armax(na, nb, nc, nk, u, y)
