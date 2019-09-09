"""
Created on Fri Sep  6 11:33:15 2019
@author: Emerson Boeira
"""
#%% Header: import python packages and libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import correlation.croscorr as corr

#from croscorr import arma_ccorr
#from croscorr import smpl_ccorr

#%% Example

Nn = 100000
sigma2e = 5
e = np.random.normal(0, np.sqrt(sigma2e), Nn)

#B = np.array([1])
#A = np.array([1])

A = np.array([1, -1.7, 0.72])
B = np.array([1, -0.95])

D = np.array([1])
C = np.array([1])

S1 = signal.TransferFunction(B, A, dt=1)
S2 = signal.TransferFunction(D, C, dt=1)

_, y = signal.dlsim(S1, e)
_, w = signal.dlsim(S2, e)

y = y[:,0]
w = w[:,0]

maxlag = 15

#%% Calculating theoretical cross covariance function
 
ryw, tau = corr.arma_ccorr(B, A, D, C, sigma2e, maxlag)

#%% Calculating sample-based cross covariance with np.cov()

ryw2, tau2 = corr.smpl_ccorr(y, w, maxlag)

#%% Graphics

plt.figure()
plt.stem(tau, ryw)
plt.stem(tau2, ryw2, 'r--')
plt.grid(True)
plt.xlabel(r'$\tau$ (samples)')
plt.ylabel(r'$r_{yy}(\tau)$')
plt.legend(('Red = Theoretical','Blue = Sampled-based'))
#plt.savefig('ex_ccorr.eps')
plt.show()