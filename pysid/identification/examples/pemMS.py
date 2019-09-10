"""
Created on Wed Apr 24 10:23:22 2019

In this example it is used the prediction error method to estimate a general
MISO black box transfer function system given as:
    A(q)y(t) = (B(q)/F(q))u(t) + (C(q)/D(q))e(t)

@author: edumapurunga
"""
#%% Example 1: Everythning is unknown
#Import Libraries
from numpy import array, absolute, convolve, concatenate, cov, diag, sqrt, \
zeros, mean
from numpy.linalg import norm
from numpy.random import random, randn #To generate the experiment
from scipy.signal import lfilter       #To generate the data
from pysid import pem                  #To estimate an arx model
#True System
#Number of inputs
nu = 2
#Number of outputs
ny = 1
#Orders
na = 1       #This variable must be (ny x ny)
nb = [1, 1]  #This variable must be (ny x nu)
nc = 2       #This variable must be (ny x 1)
nd = 2       #This variable must be (ny x 1)
nf = [1, 1]  #This variable must be (ny x nu)
nk = [1, 1]  #This variable must be (ny x nu)
#with the following true parameters
Ao = [1,  0.9]
Fo = [[1,  0.6], [1, 0.77]]
Bo = [[0, 0.5, 0.1], [0, 0.3, -0.9]]
Co = [1, 0.8, 0.2]
Do = [1, -1.6, 0.64]
#True parameter vector
thetao = [-1.2, 0.36, 0.5, 0.1, 0.8, 0.2, -1.6, 0.64]
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
AFo1 = convolve(Ao, Fo[0])
AFo2 = convolve(Ao, Fo[1])
ADo = convolve(Ao, Do)
#Calculate the y through S (BJ: G(q) = B(q)/F(q) and H(q) = C(q)/D(q))
y = lfilter(Bo[0], AFo1, u[:,0:1], axis=0) + lfilter(Bo[1], AFo2, u[:,1:2], axis=0) + lfilter(Co, ADo, e, axis=0)
#Initial Guess
A = [1, 0.8]
B = [[0, 0.45, 0.15], [0, 0.22, -0.77]]
C = [1, 0.77, 0.22]
D = [1, -1.55, 0.60]
F = [[1, 0.5], [1, 0.66]]
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
