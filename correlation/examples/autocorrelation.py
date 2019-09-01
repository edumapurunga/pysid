# -*- coding: utf-8 -*-
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

#%% Examples of usage of our functions

Nn = 10000
sigma2e = 1
e = np.random.normal(0, np.sqrt(sigma2e), Nn)

c1 = np.array([0.5])
c2 = np.array([1, -0.3])
c = 0.5 * np.convolve(c1, c2)
a1 = np.array([1, -0.8])
a2 = np.array([1, -0.95])
a = np.convolve(a1, a2)

#c = np.array([0.5])
#a = np.array([1, -0.8])

# IMPORTANT: if the numerator of the transfer function is 1, define it as num=[1], instead of num=[0,1]
# otherwise the polynomial's order computation will be wrong

y = signal.lfilter(c, a, e)

maxlags = 10

# comparing different ways to calculate the autocorrelation
# sample-based autocorrelation using our smpl_acorr() function
ryy, ty = smpl_acorr(y, maxlags)
# soderstrom algorithms to compute the theoretical autocorrelation
ryy2, ty2 = arma_acorr(c, a, maxlags)
# sample-based autocorrelation using the np.correlate() numpy function
ryy3 = np.correlate(y, y, 'full')
ryy3 = ryy3[Nn - maxlags - 1 : Nn + maxlags]

#%% Graphics

plt.figure()
plt.stem(ty2, ryy2, use_line_collection=True)
plt.stem(ty, ryy, 'r--', use_line_collection=True)
#plt.stem(ty, ryy3/(Nn-1), 'b--', use_line_collection=True))
plt.grid(True)
plt.xlabel("lag (samples)")
plt.show()