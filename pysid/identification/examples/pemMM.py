"""
Created on Wed Apr 24 10:23:22 2019

In this example it is used the prediction error method to estimate a general
MIMO black box transfer function system given as:
    A(q)y(t) = (B(q)/F(q))u(t) + (C(q)/D(q))e(t)

@author: edumapurunga
"""
#%% Example 1: Everythning is unknown
#Import Libraries
from numpy import array, absolute, convolve, concatenate, cov, diag, dot, sqrt,\
zeros, mean
from numpy.linalg import norm
from numpy.random import random, randn #To generate the experiment
from scipy.signal import lfilter       #To generate the data
from pysid import pem                  #To estimate an arx model
#True System
#Number of inputs
nu = 2
#Number of outputs
ny = 2
#Orders
na = [[1, 1], [1, 1]]   #This variable must be (ny x ny)
nb = [[1, 1], [1, 1]]   #This variable must be (ny x nu)
nc = [2, 2]             #This variable must be (ny x 1)
nd = [2, 2]             #This variable must be (ny x 1)
nf = [[1, 1], [1, 1]]   #This variable must be (ny x nu)
nk = [[1, 1], [1, 1]]   #This variable must be (ny x nu)
#with the following true parameters
Ao = array([[[1,  0.9], [0, -0.04]], [[0, 0.03], [1, 0.80]]], ndmin=2)
Fo = array([[[1,  0.6], [1,  0.77]], [[1, 0.66], [1, 0.88]]], ndmin=2)
Bo = array([[[0, 0.5, 0.1], [0, 0.3, -0.9]], [[0, 0.23, -0.11], [0, 0.43, -0.66]]], ndmin=2)
Co = array([[1, 0.8, 0.2], [1, 0.434, 0.202]], ndmin=1)
Do = array([[1, -1.313, 0.344], [1, -1.22, 0.676]], ndmin=1)
#Generate the experiment
#The true system is generates by the following relation:
# S: y(t) = Go(q)*u(t) + Ho(q)*e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 600
#Take u as uniform
u = -sqrt(3) + 2*sqrt(3)*random((N, nu))
#Generate gaussian white noise with standat deviation 0.1
e = 0.001*randn(N, ny)
#Shorthand
#AFo11 = convolve(Ao, Fo[0])
#AFo12 = convolve(Ao, Fo[1])
#AFo21 = convolve(Ao, Fo[2])
#AFo22 = convolve(Ao, Fo[3])
#ADo1 = convolve(Ao[0], Do[0])
#ADo2 = convolve(Ao[2], Do[1])
#Calculate the y through S (BJ: G(q) = B(q)/F(q) and H(q) = C(q)/D(q))
y1 = zeros((N, 1))
y2 = zeros((N, 1))
#Auxiliary Variables
v1 = lfilter(Co[0], Do[0], e[:, 0:1], axis=0)
v2 = lfilter(Co[1], Do[1], e[:, 1:2], axis=0)
w11 = lfilter(Bo[0, 0], Fo[0, 0], u[:, 0:1], axis=0)
w12 = lfilter(Bo[0, 1], Fo[0, 1], u[:, 0:1], axis=0)
w21 = lfilter(Bo[1, 0], Fo[1, 0], u[:, 1:2], axis=0)
w22 = lfilter(Bo[1, 1], Fo[1, 1], u[:, 1:2], axis=0)
for i in range(2, N):
    y1[i] = -dot(Ao[0, 0][1:3], y1[i-1:i][::-1]) - dot(Ao[0, 1][1:3], y2[i-1:i][::-1])\
          + w11[i] + w12[i] + v1[i]
    y2[i] = -dot(Ao[0, 1][1:3], y1[i-1:i][::-1]) - dot(Ao[1, 1][1:3], y2[i-1:i][::-1])\
          + w21[i] + w22[i] + v2[i]
y = concatenate((y1, y2), axis=1)
#Initial Guess
A = [[[1,  0.89], [0, -0.03]], [[0, 0.02], [1, 0.78]]]
F = [[[1,  0.58], [1,  0.71]], [[1, 0.65], [1, 0.86]]]
B = [[[0, 0.45, 0.15], [0, 0.33, -0.87]], [[0, 0.213, -0.105], [0, 0.46, -0.67]]]
C = [[1, 0.89, 0.17], [1, 0.991, 0.1]]
D = [[1, -1.55, 0.61], [1, -1.2, 0.678]]
#A = [1, 0.4]
#B = [[0, 0.3, 0.2], [0, 0.5, -0.6]]
#C = [1, 0.6, 0.4]
#D = [1, 0.6, 0.8]
#F = [[1, 0.1], [1, 0.4]]
#Estimate the model and get only the parameters
Ah, Bh, Ch, Dh, Fh = pem(A, B, C, D, F, u, y)
#%% Example 2: There are some known parameters
"""
    Now consider the situation where some parameters may be previously known.
    We can use this information in the prediction error method
"""
#Define what parameters are known by using a mask, you must indicate that a
#parameter is known by sending a 1 in the right position, while 0 indicates
#otherwise
#mask: Consider that we know both b1 = 0.5 and a1 = 0.6
#The mask should have the same size of A, B, C, D, F in the respective arrays
#Mask for polynomial A
Amask = [0, 1]
#Mask for polynomials B
Bmask = [[0, 1, 0],[0, 0, 0]]
#Mask for polynomial C
Cmask = [0, 0, 0]
#Mask for polynomial D
Dmask = [0, 0, 0]
#Mask for polynomials F
Fmask = [[0, 0], [0, 0]]
mu = [Amask, Bmask, Cmask, Dmask, Fmask]
A[1] = 0.9
B[0][1] = 0.5
Ah2, Bh2, Ch2, Dh2, Fh2 = pem(A, B, C, D, F, u, y, mu)
#%% Monte Carlo Comparison
#Number of Runs
MC = 100
#Number of parameters
d = 11
#Storage Variables
Pemu = zeros((d, MC))
Pemk = zeros((d-2, MC))
#Monte Carlo
for mc in range(0, MC):
    #New Noise Realization
    e = 0.01*randn(N, 1)
    #New Data
    y = lfilter(Bo[0], AFo1, u[:,0:1], axis=0) + lfilter(Bo[1], AFo2, u[:,1:2], axis=0) + lfilter(Co, ADo, e, axis=0)
    #Estimate all parameters
    A1, B1, C1, D1, F1 = pem(A, B, C, D, F, u, y)
    #Keep the parameters
    p = concatenate((A1[1:], B1[0][1:], B1[1][1:], C1[1:], D1[1:], F1[0][1:], F1[1][1:]))
    Pemu[:, mc] = p
    #Estimate just some parameters: b1 and a1 is known
    A1, B1, C1, D1, F1 = pem(A, B, C, D, F, u, y, mu)
    #Keep the parameters
    p = concatenate((B1[0][2:], B1[1][1:], C1[1:], D1[1:], F1[0][1:], F1[1][1:]))
    Pemk[:, mc] = p
#%% Results
#Mean
B1m = mean(Pemu, axis=1)
B2m = mean(Pemk, axis=1)
#Bias
#True Parameter vector
thetao1 = concatenate((Ao[1:], Bo[0][1:], Bo[1][1:], Co[1:], Do[1:], Fo[0][1:], Fo[1][1:]))
thetao2 = concatenate((Bo[0][2:], Bo[1][1:], Co[1:], Do[1:], Fo[0][1:], Fo[1][1:]))
bB1 = absolute(thetao1-B1m)
bB2 = absolute(thetao2-B2m)
#Covariance
covB1 = cov(Pemu)
covB2 = cov(Pemk)
#Variance of the parameters estimates
varB1 = diag(covB1)
varB2 = diag(covB2)
#Norm of the variance
nvB1 = norm(varB1)
nvB2 = norm(varB2)
#Print!
print("Pem(unk) = %5.7f, Pem(kno) = %5.7f" %(nvB1, nvB2))
