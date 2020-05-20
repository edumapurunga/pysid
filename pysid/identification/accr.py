# -*- coding: utf-8 -*-
"""
This module provides tools for accuracy analysis of parameter estimators.

author: @edumapurunga
"""
#%% Import Necessary libraries
from numpy import block, eye, kron, shape, trace, vstack, zeros
from numpy.linalg import inv
from scipy.linalg import toeplitz, solve_discrete_are, solve_discrete_lyapunov
# Internal imports
from .pemethod import *
from .ivmethod import *
from .tseries import *
#%% Define functions to call
__all__ = ['crlbss', 'crlbarma']

def crlbss(F, C, R1, R2, R12, nt, Fis, Cis, R1is, R2is, R12is):
    """
    Compute the Cramer-Rao Lower bound matrix for a linear state space
    system of the form:
        x(k+1) = F x(k) + v(k)
        y(k)   = C x(k) + e(k)

    References
    ----------
    [1] Törsten Söderström, on Computing the Cramer-Rao Bound and Covariance
    Matrices for PEM Estimates in Linear State Space Models. IFAC Proceeding
    Volumes, v. 39, n 1, p. 600-605, 2006.

    Parameters
    ----------
    F : numpy.ndarray
        Dynamic Matrix n x n matrix.
    C : numpy
        Output Matrix, p x n matrix.
    R1 : numpy
        Covariance Matrix of [v(k) ].
    R12: numpy.ndarray
        Cross Covariance of E[v(k) e(k)].
    R2 : numpy.ndarray
        Covariance of e(k)
    nt : integer
        Number of unknown Parameters
    Fis: list
        List containing the derivatives of F with repesct to the parameters
    Cis: list
        List containing the derivatives of C with repesct to the parameters
    R1is: list
        List containing the derivatives of R1 with repesct to the parameters
    R12is: list
        List containing the derivatives of R12 with repesct to the parameters
    R2is: list
        List containing the derivatives of R2 with repesct to the parameters

    Returns
    -------
    CRLB: numpy.ndarray
        The Cramer Rao Lower Bound.

    """
    # Find the dimenson of F, number of nodes/states
    n = shape(F)[0]
    p = shape(C)[0]
    # Find the Innovations Representation using the Kalman Filter
    P = solve_discrete_are(F.T, C.T, R1, R2, s=R12)
    Q = C @ P @ C.T + R2
    S = inv(Q)
    K = (F @ P @ C.T + R12) @ S
    # Sensitivity Matrices
    Pis = []
    Qis = []
    Kis = []
    # Solve Discrete Lyapunov
    for i in range(nt):
        # Auxiliary
        aux = (R1is[i] - K@R12is[i].T - R12is[i]@K.T + K@R2is[i]@K.T) + (Fis[i]-K@Cis[i])@P@(F-K@C).T + (F - K@C)@P@(Fis[i] - K@Cis[i]).T
        # Solve Discrete Lyapynov
        Pis.append(solve_discrete_lyapunov((F - K@C), aux))
        # Update Q
        Qis.append(Cis[i]@P@C.T + C@P@Cis[i].T + C@Pis[i]@C.T + R2is[i])
        # Update K
        Kis.append((Fis[i]-K@Cis[i])@P@C.T@S + (F - K@C)@Pis[i]@C.T@S + R12is[i]@S + (F - K@C)@P@Cis[i].T@S - K@R2is[i]@S)
    # Form the augmented state space system
    Fb = vstack(tuple(Fis))
    Cb = vstack(tuple(Cis))
    Kb = vstack(tuple(Kis))
    # stack K*Ci and KiC
    KbCis = []
    KCbis = []
    for i in range(nt):
        KbCis.append(Kis[i].dot(C))
        KCbis.append(K.dot(Cis[i]))
    # Stack
    KbCi = vstack(tuple(KbCis))
    KCbi = vstack(tuple(KCbis))
    # State Matrices
    # This is probably wrong for  "Fc - Kb.dot(C)- K.dot(Cb)"
    cF = block([[F, zeros((n, n)), zeros((n, n*nt))], [K.dot(C), F-K.dot(C), zeros((n, n*nt))], [Kb.dot(C), Fb-Kb.dot(C)-KCbi, kron(eye(nt), F-K.dot(C))]])
    cK = block([[eye(n), zeros((n, p))], [zeros((n, n)), K], [zeros((n*nt, n)), Kb]])
    cC = block([[zeros((nt*p, n)), -Cb, kron(-eye(nt), C)]])
    # Covariance of the augmented systems
    covv = block([[R1, R12], [R12.T, R2]])
    # Solve the Lyapunov Equation for the augmented state space system
    cP = solve_discrete_lyapunov(cF, cK.dot(covv.dot(cK.T)))
    # Get  E[psispi] as:
    PP = cC.dot(cP.dot(cC.T))
    # Compute Pbar and EpsiWpsi
    Pb = zeros((nt, nt))
    pwp = zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            # Pbar
            Pb[i, j] = 0.5*trace(S@Qis[i]@S@Qis[j])
            # Weighted version of Epsipsi
            pwp[i, j] = trace(PP[j, i]*S)
    # Compute the CRLB
    return inv(pwp + Pb)

def crlbarma(A, C, sig):
    """
    Returns the Cramer-Rao Lower Bound for an ARMA process:
        y(t) = C(q)/A(q) e(t)

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    C : TYPE
        DESCRIPTION.
    sig : TYPE
        DESCRIPTION.

    Returns
    -------
    CRLB : TYPE
        DESCRIPTION.

    References
    ----------
    [1]

    """
    # Verify max
    n = len(A)-1
    m = len(C)-1
    # Max
    df = n-m
    # Fill with zeros
    if df < 0:
        q = m
        A += [0]*(df*-1)
    else:
        q = n
        C += [0]*(df)
    # Build the Matrices of Parameters
    A1 = toeplitz(A[0:-1], [A[0]] + [0]*(q-1))
    A2 = toeplitz(A[1:][::-1], [A[-1]] + [0]*(q-1))
    C1 = toeplitz(C[0:-1], [C[0]] + [0]*(q-1))
    C2 = toeplitz(C[1:][::-1], [C[-1]] + [0]*(q-1))
    # Explicit Formulae
    bRxx = inv(A1@A1.T - A2@A2.T)
    bRzz = inv(C1@C1.T - C2@C2.T)
    bRzx = inv(A1@C1.T - C2@A2.T)
    # Recovery of the CRLB
    M = block([[bRxx[0:n, 0:n], -bRzx.T[0:n, 0:m]], [-bRzx[0:m, 0:n], bRzz[0:m, 0:m]]])
    CRLB = inv(M)
    return CRLB

# Kalman Filtering
def kalman(A, B, C, D):
    return NotImplemented
