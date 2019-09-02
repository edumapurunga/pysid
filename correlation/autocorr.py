#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:13:14 2019
@author: Emerson Boeira
"""
#%% Header: importing python packages and libraries

import numpy as np  # important package for scientific computing

#%% Functions that calculate sample and theoric autocorrelation of ARMA processes

def smpl_acorr(y, maxlag):
    # Description to help the user
    """Function that calculates the sample-based autocorrelation of the signal y(t), using the numpy.cov() function.
    The advantage os this function is that it's faster and more efficient that numpy.correlate().
    
    Inputs: y, maxlag
    Outputs: ryy, tau
    
    Inputs description:
        y: observed signal
        maxlag: maximum lag that will be considered on the computation of the autocorrelation (from -maxlag to +maxlag)
            
    Outputs description:
        ryy: the autocorrelation function, calculated based on the observation of y(t);
        tau: the lag interval considered. It has the same size as ryy."""
    
    # calculating the size of tau
    N = 2 * maxlag + 1
    # assembling tau with linspace
    tau = np.linspace(- maxlag, maxlag, N)  # linspace(start, stop, numberofpoints)
    
    # preallocating the autocorrelation function vector
    ryy = np.zeros((maxlag + 1))
    
    # calculating the autocorrelation for tau=0 using the cov function
    ryy[0] = np.cov(y[1:], y[:-1])[0][0]

    # calculating the correlation function
    for l in range(1, maxlag + 1):
        ryy[l] = np.cov(y[l:], y[:-l])[0][1]
    
    # using the flip operation to return a vector that represents the autocorrelation from -maxlag to +maxlag
    ryyf = np.flip(ryy[1:])
    ryyc = np.concatenate((ryyf, ryy))
    
    # returns the sample-based autocorrelation and the lag vector
    return ryyc, tau

def arma_acorr(C, A, var, maxlag):
    # Description to help the user
    """Function that calculates the theoretical autocorrelation function of an ARMA process:
        A(q)y(t)=C(q)e(t), where
        A(q) and C(q) are defined as polinomials on q (instead of q^{-1})
        e(t) is a white noise sequence with unitary variance: E[e(t)e(t)] = 1
    
    Inputs: C, A, maxlag
    Outputs: ryy, tau
    
    Inputs description:
        A = vector that contains the coefficients of A(q)
        C = vector that contains the coefficients of C(q)
        maxlag: maximum lag that will be considered on the computation of the autocorrelation (from -maxlag to +maxlag)
            
    Outputs description:
        ryy: the autocorrelation function, calculated based on Soderstrom's algorithm
        tau: the lag interval considered. It has the same size as ryy."""
    
    # order of A(q)
    n = A.shape[0] - 1
    # order of C(q)
    m = C.shape[0] - 1

    # calculating C(q^-1)
    cm = np.flip(C)

    # making C(q) and C(q^-1) with the same base
    zc = np.zeros((m))
    C = np.concatenate((C, zc))
    cm = np.concatenate((zc, cm))

    # calculating B(q)
    B = np.convolve(C, cm)
    # fixing the dimensions after the convolve
    B = B[m : 3 * m + 1]
    # making B(q) with the same shape as A(q)
    zb = np.zeros((n - m))
    B = np.concatenate((zb, B))
    B = np.concatenate((B, zb))
    # taking only the redundant part of B(q)
    Bn = B[0 : n + 1]
    # fliping the bn vector to find D(q)
    Bn = np.flip(Bn)
    
    # preallocating the A1 and A2 matrices
    A1 = np.zeros((n + 1, n + 1))
    A2 = np.zeros((n + 1, n + 1))

    # assembling A1 and A2
    for k in range(0, n + 1):
        A1[k][0 : n + 1 - k] = A[k : n + 1]
        A2[k][k : n + 1] = A[0 : n + 1 - k]

    # assembling the matrix "calligraphic A" - to avoid redundance we'll call it Acal
    Acal = A1 + A2

    # finding the polynomial D(q)
    D = np.linalg.solve(Acal, Bn)
    
    # using the function coeff to calculate the coefficients of the correlation based on A(q) and D(q)
    ryy, tau = coeff(A, D, maxlag)
    
    # scaling the correlation function with the variance of e(t)
    ryy = var * ryy
    
    # returning the theoretical correlation and the lag vector
    return ryy, tau
    
def coeff(A, D, maxlag):
     # Description to help the user
    """Function that calculates the coefficients of the theoretical correlation, based on the A(q) and D(q) polynomials,
    from -maxlag to +maxlag.
    
    Inputs: A, D, maxlag
    Outputs: ryy, tau
    
    Inputs description:
        A = vector that contains the coefficients of A(q)
        D = vector that contains the coefficients of D(q)
        maxlag: maximum lag that will be considered on the computation of the autocorrelation (from -maxlag to +maxlag)
            
    Outputs description:
        ryy: the autocorrelation function, calculated based on Soderstrom's algorithm
        tau: the lag interval considered. It has the same size as ryy."""
    
    # preallocating the correlation function vector
    ryy = np.zeros((maxlag + 1))

    # computing the order of A(q)
    n = A.shape[0] - 1
    
    # calculating the first coefficient of the correlation function
    ryy[0] = 2 * D[0]
    # calculating the second coefficient of the correlation function
    ryy[1] = D[1] - A[1]*D[0]

    # loop that calculates the other coefficients
    for k in range(2, maxlag + 1):
        Sr = 0
        Sk = 0
        if k <= n:
            for j in range(1, k):
                Sk = Sk + A[j]*ryy[k - j]
                ryy[k] = D[k]-A[k]*D[0]-Sk
        else:
            for j in range(1, n + 1):
                Sr = Sr + A[j]*ryy[k - j]
                ryy[k] = -Sr
                
    # using the flip operation to return a vector that represents the autocorrelation from -maxlag to +maxlag
    ryyf=np.flip(ryy[1:])
    ryyc = np.concatenate((ryyf, ryy))
        
    # calculating the size of tau
    N = 2 * maxlag + 1
    # assembling tau with linspace
    tau = np.linspace(- maxlag, maxlag, N)  # linspace(start, stop, numberofpoints)
    
    # returning
    return ryyc, tau