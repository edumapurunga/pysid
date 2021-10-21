# encoded utf-8
"""
This module provides tools for model determination. That includes the
traditional information criteria, such as Akaike information criterion
(AIC), BIC, Final predictor error (FPE) and other tools to determine
model orders.

author: @edumapurunga
"""
#%%
from numpy import dot, empty, log, amin, where
from scipy.signal import lfilter
from numpy.random import rand, randn

from pemethod import arx

def aicarx(na_max, nb_max, nk_max, u, y):
    """
    author: @lima84

    Estimates ARX model based on Akaike's Information Criterion (AIC) given
    the upper limits for the polynomial orders (na_max, nb_max, nk_max) and
    a pair of input-output data vectors (u, y). Returns the lowest AIC cost
    and the best fitting A(q) and B(q) polynomials for the ARX model: 
        A(q)y(t) = B(q)u(t) + e(t),

	Parameters
	----------
	na_max : int
		maximum value for the na parameter -- na = [1, 2, ..., na_max]
	nb_max : int
		maximum value for the na parameter -- nb = [0, 1, ..., nb_max]
	nk_max : int
		maximum value for the na parameter -- nk = [0, 1, ..., nk_max]
    u : ndarray
        input data array
    y : ndarray
        output data array
	Returns
	-------
	A : ndarray
		Array containing the A(q) polynomial
	B : ndarray
		Array containing the B(q) polynomial
    J_aic : int
        AIC cost function value using A(q) and B(q)
    """
    
    A_aic = empty((na_max,nb_max,nk_max), dtype='object')
    B_aic = empty((na_max,nb_max,nk_max), dtype='object')
    J_aic = empty((na_max,nb_max,nk_max), dtype='object')

    for na in range(1,na_max+1):
        for nb in range(0,nb_max+1):
            for nk in range(0,nk_max+1):
                # Computes ARX polynomials for current (na, nb, nk)
                A,B = arx(na, nb, nk, u, y)

                # Array-list magic for lfilter 
                A = A.tolist()[0][0]
                B = B.tolist()[0][0]

                # Computes e(t) = A(na,nb,nk,q)y(t) - B(na,nb,nk,q)u(t)
                e = lfilter(A, [1], y, axis=0) - lfilter(B, [1], u, axis=0)

                N = len(e)
                p = len(A) + len(B)

                # Computes the cost function
                J = (1/N) * dot(e.T,e)[0][0]

                # Add current polynomials to their respective matrix
                A_aic[na-1,nb-1,nk-1] = A
                B_aic[na-1,nb-1,nk-1] = B

                # Computes AIC cost function
                J_aic[na-1,nb-1,nk-1] = N * log(J) + 2*p

    # Finds the lowest cost estimate indices
    min_index = where(J_aic == amin(J_aic))
    A, B, J_aic = A_aic[min_index],B_aic[min_index],J_aic[min_index]
    return [A, B, J_aic]
