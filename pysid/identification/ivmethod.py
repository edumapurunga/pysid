#This modules provides instrumental variables methods for discrete-time linear
#models
"""
Module for Instrumental Variables methods

@author: edumapurunga
"""

#%% imports
from numpy import arange, array, append, copy, count_nonzero, ones,\
delete, dot, empty, sum, size, amax, matrix, concatenate, shape, zeros, kron,\
eye, reshape, convolve, sqrt, where, nonzero, correlate, equal, ndarray, pi, \
absolute, exp, log, real
from scipy.linalg import qr, solve, toeplitz
from numpy.linalg import matrix_rank
from scipy.signal import lfilter, periodogram
from scipy.optimize import leastsq, least_squares
import numpy.fft as fft
#%% functions
__all__ = ['iv']
#%% Implementations
def iv(na, nb, nk, u, y, y2):
    '''
    :param na: number of zeros from A;
    :param nb: number of poles from B;
    :param u: input signal;
    :param y: output signal;
    :param y2:
    :param nk: output signal delay;
    :return: coefficients of A and B in this order;
    '''
    # Number of samples
    N = size(y)
    # Vetors u, y and y2 must have same amount of samples
    if (N != size(u)) or (N != size(y2)):
        raise ValueError('Y, Y2 and U must have same length!')
    # Number of coefficients to be estimated
    # (a_1, a_2, a_3,..., a_na, b_0, b_1, b_2, b_nb)
    M = na + nb + 1
    # Delay maximum needed
    n_max = amax([na, nb + nk])
    # In order to estimate the coeffiecients, we will need to delay the samples.
    # If the maximum order is greater than the number of samples,
    # then it will not be possible!
    if not (N - n_max > 0):
        raise ValueError('Number of samples should be greater' &
                         'than the maximum order!')
    # Build matrix phi in which will contain y and u shifted in time
    phi = zeros((N - n_max, M))
    # Build matrix csi in which will contain y2 and u shifted in time
    csi = zeros((N - n_max, M))
    k = 0
    # Fill phi/csi with y/y2 shifted in time from 0 to nb
    for i in range(1, na + 1):
        phi[:, k] = -y[n_max - i:N - i]
        csi[:, k] = -y2[n_max - i:N - i]
        k = k + 1
    # Fill phi/csi with u shifted in time from 0 to nb
    for i in range(nk, nb + nk + 1):
        phi[:, k] = u[n_max - i:N - i]
        csi[:, k] = u[n_max - i:N - i]
        k = k + 1
    # Crop y from n_max to N
    y = y[n_max:N]
    # Find theta
    R = dot(csi.T, phi)
    # If the experiment is not informative:
    if (matrix_rank(R) < M):
        raise ValueError('Experiment is not informative')
    S = dot(csi.T, y)
    theta = solve(R, S)
    # Split theta in vectors a and b
    a = theta[0:na]
    b = theta[na:na + nb + 1]
    return [a, b]

#%% Auxiliary Functions
