"""
Created on Thu Aug 29 14:54:13 2019
@author: Emerson Boeira
"""
#%% Header: import python packages and libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics

from autocorr import smpl_acorr
from autocorr import arma_acorr

#%% Examples

Nn = 100000
sigma2e = 1
e = np.random.normal(0, np.sqrt(sigma2e), Nn)

A = np.array([1, -0.9])
C = np.array([1])

S1 = signal.TransferFunction(C, A, dt=1)

_, y = signal.dlsim(S1, e)

y = y[:,0]

maxlag = 15

#%% Calculating theoretical autocovariance function
 
ryy, tau = arma_acorr(C, A, sigma2e, maxlag)

#%% Calculating sample-based cross covariance with np.cov()

ryy2, tau2 = smpl_acorr(y, maxlag)

#%% Graphics

plt.figure()
plt.stem(tau2, ryy2)
plt.stem(tau, ryy, 'r--')
plt.grid(True)
plt.xlabel(r'$\tau$ (samples)')
plt.show()