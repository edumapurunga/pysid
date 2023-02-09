#Time Series models
"""
Module for estimate time series models (AR, MA, ARMA)

author: @edumapurunga
"""

# Imports
from numpy import arange, array, append, copy, count_nonzero, ones,\
delete, dot, empty, sum, size, amax, matrix, concatenate, shape, zeros, kron,\
eye, reshape, convolve, sqrt, where, nonzero, correlate, equal, ndarray, pi, \
absolute, exp, log, real
from scipy.linalg import qr, solve, toeplitz
from numpy.linalg import matrix_rank
from scipy.signal import lfilter, periodogram
from scipy.optimize import leastsq, least_squares
import numpy.fft as fft
# Internal imports
from .solvers import ls, levinson, burg
from ..io.check import chckin
# functions
__all__ = ['ar', 'arma', 'ma']
# implementations
def ar(na, y, md = 'yw'):
    """
    This function estimate a AR model based on output data provided in the
    vector y. This particular function returns the polynomial A(q) from the
    following AR model:
        A(q)y(t) = e(t)
    Inputs:
    Outputs:
    """
    na, _, _, _, _, _, _, y = chckin(na, [], [], [], [], [], y, y)
    Ny, ny = shape(y)
    na = na.item()
    # Yule-Walker solution
    if md == 'yw':
        # Find the correlation
        R = correlate(y.reshape(Ny,), y.reshape(Ny,), "full")
        # Biased Correlaton
        R = R/Ny
        R = R[0:na+1]
        A = levinson(R, na)
        A = A[-1]
    # Burg Algorithm
    elif md == 'burg':
        A = burg(y, na)
    # Prediction Error algorithm
    elif md == 'pem':
        # Define the prediction error
        def pe(theta, na, y):
            return lfilter(append([1], theta[0:na]), [1], y, axis=0)
        # Least Squares Initialization
        thetai = ls(na, -1, 0, y, y)[0]
        sol = least_squares(pe, thetai, gtol=1e-15, args=(na, y.reshape((Ny))))
        theta = sol.x
        A = append([1], theta)
    return A

def arma(na, nc, y, md='pem'):
    """
    This functions estimates the parameters of an ARMA model defined as:
        A(q)y(t) = C(q)e(t)
    """
    na, _, nc, _, _, _, _, y = chckin(na, [], nc, [], [], [], y, y)
    # size
    Ny, ny = shape(y)
    # Hannan-Rissanen Algorithm
    if md == 'hannan':
        # Step 1: Estimate a high order AR
        n = 50
        Ar = ar(n, y, md='burg')
        ehat = lfilter(Ar, [1], y, axis=0)
        # Step 2: Estimate an initial ARMA model
        A1, B1 = ls(na, nc-1, 1, ehat, y)
        # Step 3: Reestimate based on an approximation of ML
        etil = lfilter(append([1], A1), [1], y, axis=0) - lfilter(append([0], B1), [1], ehat, axis=0)
        # Predictors Inputs
        napa = lfilter([1], append([1], B1), y, axis=0)
        qsi = lfilter([1], append([1], B1), etil, axis=0)
        # Predicted Output
        Y = etil + napa - qsi
        u = concatenate((-napa, qsi), axis=1)
        _, B = arx(0, [[na-1, nb]], [1, 1], u, Y)
        A = B[0, 0][1:]
        B = B[0, 1][1:]
        A = append([1], A)
        B = append([1], B)
    # PEM algorithm
    if md == 'pem':
        # Define the prediction error
        def pe(theta, na, nc, y):
            return lfilter(append([1], theta[0:na]), append([1], theta[na:]), y, axis=0)
        # Least Squares Initialization
        n = 50
        Ar = ar(n, y, md='burg')
        ehat = lfilter(Ar, [1], y, axis=0)
        A1, B1 = ls(na, nc-1, 1, ehat, y)
        thetai = concatenate((A1, B1))
        sol = least_squares(pe, thetai, gtol=1e-15, args=(na, nc, y.reshape((Ny))))
        theta = sol.x
        A = append([1], theta[0:na])
        B = append([1], theta[na:])
    return [A, B]

def ma(nc, y, md='durbin'):
    """
    This function estimates the parameters of a moving average model in the form:
        y(t) = C(q)e(t)
    """
    nc = array(nc)
    y = array(y)
    Ny, ny = shape(y)
    # Durbin Method
    if md == 'durbin':
        # Estimate a high order AR
        n = 2*nc.item()
        Ar = ar(n, y, 'burg')
        # Estimate the innovations
        ehat = lfilter(Ar, [1], y, axis=0)
        v = y - ehat
        B = ls(0, nc-1, 1, ehat, v)[1]
        C = append([1], B)
    # Vocariance recursion method
    if md == 'vrm':
        Psi = zeros((Ny,))
        Psi2= zeros((Ny,))
        t = arange(0, Ny)
        wk = 2*t*pi/Ny
        # Estimate of the periodogram
        for p in range(0, Ny):
            Psi[p] = 1/Ny*absolute(sum(y*exp(-1j*wk[p]*t)))**2
        Psi2 = periodogram(y, scaling='spectrum', axis=0)[1]
        # Estimate the vocariances
        c = zeros((int(Ny/2),))
        c2 = absolute(fft.ifft(log(Psi2), axis=0))**2
        c2 = c2.reshape((101,))
        for k in range(0, int(Ny/2)):
            #TODO: Verify how to compute the ceptrum
            c[k] = 1/Ny*sum(real(log(Psi)*exp(-1j*wk[k]*t)))
        # Estimate the MA parameters
        b = zeros((nc+2,))
        b[0] = 1
        for j in range(1, nc+2):
            for p in range(0, j):
                b[j]+= (j-p)*c2[j-p]*b[p]
            b[j] = b[j]/j
        C = copy(b)
    # Prediction Error Method
    if md == 'pem':
        # Define the prediction error
        def pe(theta, nc, y):
            return lfilter([1], append([1], theta[0:nc+1]), y, axis=0)
        # Estimate a high order AR
        n = 50
        Ar = ar(n, y, 'burg')
        # Estimate e hat
        ehat = lfilter(Ar, [1], y, axis=0)
        # Least Squares Initialization
        thetai = ls(0, nc-1, 1, ehat, y)[1]
        sol = least_squares(pe, thetai, gtol=1e-15, args=(nc, y.reshape((Ny))))
        theta = sol.x
        C = append([1], theta)
    return C
