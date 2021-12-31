"""
This module provides tools for model determination. That includes the
traditional information criteria, such as Akaike information criterion
(AIC), BIC, Final predictor error (FPE) and other tools to determine
model orders.
"""
#
from numpy import dot, empty, log, amin, where
from scipy.signal import lfilter

from .pemethod import arx
from ..io.check import chckin

__all__ = ['aicarx']

def aiccrit(J, N, p):
    """Retun the AIC criterion"""
    return N*log(J) + 2*p

def aicncrit(J, N, p):
    """Return the normalized AIC criterion"""
    return log(J) + 2*p/N

def aicccrit(J, N, p):
    """Return the corrected AIC criterion"""
    return N*log(J) + 2*p + 2*p*(p + 1)/(N - p - 1)

def aicarx(na_max, nb_max, nk_max, u, y, criterion='aicn'):
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
    criterion: string (optional)
        critrion to be evaluated.
    Returns
    -------
    A : ndarray
        Array containing the A(q) polynomial
    B : ndarray
        Array containing the B(q) polynomial
    J_aic : int
        AIC cost function value using A(q) and B(q)
    """
    # Check input arguments
    _, _, _, _, _, _, u, y = chckin(na_max, nb_max, 0, 0, 0, nk_max, u, y)

    # Number of samples and outputs
    N, ny = y.shape

    A_aic = empty((na_max, nb_max + 1, nk_max + 1), dtype='object')
    B_aic = empty((na_max, nb_max + 1, nk_max + 1), dtype='object')
    J_aic = empty((na_max, nb_max + 1, nk_max + 1), dtype='object')

    criteria = {'aic': aiccrit, 'aicn': aicncrit, 'aicc': aicccrit}
    crit = criteria.get(criterion)

    for na in range(1,na_max+1):
        for nb in range(0,nb_max+1):
            for nk in range(0,nk_max+1):
                # Computes ARX polynomials for current (na, nb, nk)
                A, B = arx(na, nb, nk, u, y)

                # Array-list magic for lfilter 
                A = A.tolist()[0][0]
                B = B.tolist()[0][0]

                # Computes e(t) = A(na,nb,nk,q)y(t) - B(na,nb,nk,q)u(t)
                e = lfilter(A, [1], y, axis=0) - lfilter(B, [1], u, axis=0)

                # Number of parameters
                p = na + nb + 1

                # Computes the cost function
                J = (1/N) * dot(e.T, e)[0][0]

                # Add current polynomials to their respective matrix
                A_aic[na - 1, nb, nk] = A
                B_aic[na - 1, nb, nk] = B

                # Computes AIC cost function
                J_aic[na - 1, nb, nk] = crit(J, N, p)

    # Finds the lowest cost estimate indices
    min_index = where(J_aic == amin(J_aic))

    A, B, J_aic = A_aic[min_index], B_aic[min_index], J_aic[min_index]
    return [A, B, J_aic]
