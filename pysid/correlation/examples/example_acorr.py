"""
Created on Thu Aug 29 14:54:13 2019
@author: Emerson Boeira
"""
#%% Header: import python packages and libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import correlation.autocorr as corr

#from autocorr import smpl_acorr
#from autocorr import arma_acorr

#%% Examples

Nn = 100000
sigma2e = 1
e = np.random.normal(0, np.sqrt(sigma2e), Nn)

A = np.array([1, -1.7, 0.72])
C = np.array([1, -0.95])

S1 = signal.TransferFunction(C, A, dt=1)

_, y = signal.dlsim(S1, e)

y = y[:,0]

maxlag = 15

#%% Calculating theoretical autocovariance function
 
ryy, tau = corr.arma_acorr(C, A, sigma2e, maxlag)

#%% Calculating sample-based autocovariance with np.cov()

ryy2, tau2 = corr.smpl_acorr(y, maxlag)

#%% Graphics

plt.figure()
plt.stem(tau2, ryy2)
plt.stem(tau, ryy, 'r--')
plt.grid(True)
plt.xlabel(r'$\tau$ (samples)')
plt.ylabel(r'$r_{yy}(\tau)$')
plt.legend(('Red = Theoretical','Blue = Sampled-based'))
#plt.savefig('ex_acorr.eps')
plt.show()
