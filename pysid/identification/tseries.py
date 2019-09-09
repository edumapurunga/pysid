#Time Series models
"""
Module for estimate time series models (AR, MA, ARMA)

author: @edumapurunga
"""

#%%Imports
from numpy import arange, array, append, asscalar, copy, count_nonzero, ones,\
delete, dot, empty, sum, size, amax, matrix, concatenate, shape, zeros, kron,\
eye, reshape, convolve, sqrt, where, nonzero, correlate, equal, ndarray, pi, \
absolute, exp, log, real
from scipy.linalg import qr, solve, toeplitz
from numpy.linalg import matrix_rank
from scipy.signal import lfilter, periodogram
from scipy.optimize import leastsq, least_squares
import numpy.fft as fft
#Internal imports
from .pemethod import chckin
from .pemethod import ls
#%% functions
__all__ = ['ar', 'arma', 'ma']
#%% implementations
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
    #Yule-Walker solution
    if md == 'yw':
        #Find the correlation
        R = correlate(y.reshape(Ny,), y.reshape(Ny,), "full")
        #Biased Correlaton
        R = R/Ny
        R = R[0:na+1]
        A = levinson(R, na)
        A = A[-1]
    #Burg Algorithm
    elif md == 'burg':
        A = burg(y, na)
    #Prediction Error algorithm
    elif md == 'pem':
        #Define the prediction error
        def pe(theta, na, y):
            return lfilter(append([1], theta[0:na]), [1], y, axis=0)
        #Least Squares Initialization
        thetai = ls(na, -1, 0, y, y)[0]
        sol = least_squares(pe, thetai, gtol=1e-15, args=(na, y.reshape((Ny))))
        theta = sol.x
        A = append([1], theta)
    return A

def arma(na, nb, y, md='pem'):
    """
    This functions estimates the parameters of an ARMA model defined as:
        A(q)y(t) = B(q)e(t)
    """
    y = array(y)
    if na != 0:
        na = array(na)
    if nb != 0:
        nb = array(nb)
    #size
    Ny, ny = shape(y)
    #Hannan-Rissanen Algorithm
    if md == 'hannan':
        #Step 1: Estimate a high order AR
        n = 50
        Ar = ar(n, y, md='burg')
        ehat = lfilter(Ar, [1], y, axis=0)
        #Step 2: Estimate an initial ARMA model
        A1, B1 = ls(na, nb, 1, ehat, y)
        #Step 3: Reestimate based on an approximation of ML
        etil = lfilter(append([1], A1), [1], y, axis=0) - lfilter(append([0], B1), [1], ehat, axis=0)
        #Predictors Inputs
        napa = lfilter([1], append([1], B1), y, axis=0)
        qsi = lfilter([1], append([1], B1), etil, axis=0)
        #Predicted Output
        Y = etil + napa - qsi
        u = concatenate((-napa, qsi), axis=1)
        _, B = arx(0, [[na-1, nb]], [1, 1], u, Y)
        A = B[0, 0][1:]
        B = B[0, 1][1:]
        A = append([1], A)
        B = append([1], B)
    #PEM algorithm
    if md == 'pem':
        #Define the prediction error
        def pe(theta, na, nb, y):
            return lfilter(append([1], theta[0:na]), append([1], theta[na:]), y, axis=0)
        #Least Squares Initialization
        n = 50
        Ar = ar(n, y, md='burg')
        ehat = lfilter(Ar, [1], y, axis=0)
        A1, B1 = ls(na, nb, 1, ehat, y)
        thetai = concatenate((A1, B1))
        sol = least_squares(pe, thetai, gtol=1e-15, args=(na, nb, y.reshape((Ny))))
        theta = sol.x
        A = append([1], theta[0:na])
        B = append([1], theta[na:])
    return [A, B]

def ma(nb, y, md='durbin'):
    """
    This function estimates the parameters of a moving avarage model in the form:
        y(t) = B(q)e(t)
    """
    nb = array(nb)
    y = array(y)
    Ny, ny = shape(y)
    #Durbin Method
    if md == 'durbin':
        #Estimate a high order AR
        n = 2*nb.item()
        Ar = ar(n, y, 'burg')
        #Estimate the innovations
        ehat = lfilter(Ar, [1], y, axis=0)
        v = y - ehat
        B = ls(0, nb, 1, ehat, v)[1]
        B = append([1], B)
    #Vocariance recursion method
    if md == 'vrm':
        Psi = zeros((Ny,))
        Psi2= zeros((Ny,))
        t = arange(0, Ny)
        wk = 2*t*pi/Ny
        #Estimate of the periodogram
        for p in range(0, Ny):
            Psi[p] = 1/Ny*absolute(sum(y*exp(-1j*wk[p]*t)))**2
        Psi2 = periodogram(y, scaling='spectrum', axis=0)[1]
        #Estimate the vocariances
        c = zeros((int(Ny/2),))
        c2 = absolute(fft.ifft(log(Psi2), axis=0))**2
        c2 = c2.reshape((101,))
        for k in range(0, int(Ny/2)):
            #TODO: Verify how to compute the ceptrum
            c[k] = 1/Ny*sum(real(log(Psi)*exp(-1j*wk[k]*t)))
        #Estimate the MA parameters
        b = zeros((nb+2,))
        b[0] = 1
        for j in range(1, nb+2):
            for p in range(0, j):
                b[j]+= (j-p)*c2[j-p]*b[p]
            b[j] = b[j]/j
        B = copy(b)
    #Prediction Error Method
    if md == 'pem':
        #Define the prediction error
        def pe(theta, nb, y):
            return lfilter([1], append([1], theta[0:nb+1]), y, axis=0)
        #Estimate a high order AR
        n = 50
        Ar = ar(n, y, 'burg')
        #Estimate ê
        ehat = lfilter(Ar, [1], y, axis=0)
        #Least Squares Initialization
        thetai = ls(0, nb, 1, ehat, y)[1]
        sol = least_squares(pe, thetai, gtol=1e-15, args=(nb, y.reshape((Ny))))
        theta = sol.x
        B = append([1], theta)
    return B

#%% Auxiliary functions
def levinson(R, n):
    """
    This function implements the Levinson algorithm for fast parameters computa
    tions
    """
    A = empty((n,), dtype='object')
    alfa = append(R[1], zeros((1, n)))
    E = append(R[0], zeros((1, n)))
    #Levinson Algorithm
    for i in range(0, n):
        k = -alfa[i]/E[i]
        E[i+1] = (1 - abs(k)**2)*E[i]
        if i == 0:
            Av = array([1, k])
        else:
            An = Av[1:] + k*Av[1:][::-1]
            Av = append(Av[0], An)
            Av = append(Av, k)
        if i != n-1:
            alfa[i+1] = dot(Av, R[1:i+3][::-1]) #It should be dot here
        A[i] = Av
    return A

def burg(y, n):
    #Array Everything
    y = array(y)
    n = array(n)
    #Size
    N, ny = shape(y)
    #Initialization
    fi = y[1:]
    gi = y[0:-1]
    a = array([1])
    Epsilon = zeros((n+1,))
    Epsilon[0] = dot(y.T, y)
    K = zeros((n,))
    for i in range(0, n):
        K[i] = -dot(fi.T, gi)/((dot(fi.T, fi) + dot(gi.T, gi))/2)
        a = append(a, [0]) + K[i]*append([0], a[0:][::-1])
        fin = fi + K[i]*gi
        gin = K[i]*fi + gi
        fi = fin[1:]
        gi = gin[0:-1]
        Epsilon[i+1] = (1-K[i]*K[i])
    return a
