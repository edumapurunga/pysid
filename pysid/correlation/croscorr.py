"""
Created on Thu Sep  5 14:45:46 2019
@author: Emerson Boeira
"""
#%% Header: importing python packages and libraries

import numpy as np  # important package for scientific computing

#%% Functions that calculate the sample-based and the theoretical cross correlation of ARMA processes

def smpl_ccorr(y, w, maxlag):
    """Function that calculates the sample-based cross correlation between the signals y(t) and w(t),
    using the numpy.cov() function. The advantage os this function is that it's faster and more efficient
    that numpy.correlate().
    
    Inputs: y, w, maxlag
    Outputs: ryy, tau
    
    Inputs description:
        y: observed signal
        w: observed signal
        maxlag: maximum lag that will be considered on the computation of the cross correlation (from -maxlag to +maxlag)
            
    Outputs description:
        ryw: the cross correlation function, calculated based on the observations of y(t) and w(t);
        tau: the lag interval considered. It has the same size as ryw."""
        
    # calculating the size of tau
    N = 2 * maxlag + 1
    # assembling tau with linspace
    tau = np.linspace(- maxlag, maxlag, N)  # linspace(start, stop, numberofpoints)
    
    # preallocating the positive and negative parts of the cross correlation function
    rywp = np.zeros((maxlag + 1))
    rywn = np.zeros((maxlag))

    # first element
    rywp[0] = np.cov(y[0:], w[0:])[0][1]

    # calculating the cross correlation function
    for l in range(1, maxlag + 1):
        rywn[l-1] = np.cov(y[:-l], w[l:])[0][1]
        rywp[l] = np.cov(y[l:], w[:-l])[0][1]
        
    # using the flip operation to return a vector that represents the autocorrelation from -maxlag to +maxlag
    rywn = np.flip(rywn)
    ryw = np.concatenate((rywn, rywp))
    
    # returns the sample-based cross correlation and the lag vector
    return ryw, tau

def arma_ccorr(B, A, D, C, var, maxlag):
    # Description to help the user
    """Function that calculates the theoretical cross correlation function of two ARMA processess:
        A(q)y(t)=B(q)e(t), and
        C(q)w(t)=D(q)e(t), where
        A(q), B(q), C(q) and D(q) are defined as polinomials on q (instead of q^{-1})
        e(t) is a white noise sequence, commom with both processes with variance: E[e(t)e(t)] = var
    
    Inputs: B, A D, C, var, maxlag
    Outputs: ryw, tau
    
    Inputs description:
        B = vector that contains the coefficients of B(q) on a "q basis"
        A = vector that contains the coefficients of A(q) on a "q basis"
        D = vector that contains the coefficients of D(q) on a "q basis"
        C = vector that contains the coefficients of C(q) on a "q basis"
        var = variance of e(t)
        maxlag = maximum lag that will be considered on the computation of the cross correlation (from -maxlag to +maxlag)
            
    Outputs description:
        ryw: the cross correlation function, calculated based on Soderstrom's algorithm
        tau: the lag interval considered. It has the same size as ryw."""
        
    # transforms A(q) and B(q) to the same polynomial base
    # order of A(q)
    n = A.shape[0] - 1
    # order of B(q)
    nb = B.shape[0] - 1
    # add zeros to B(q)
    ndif = n - nb
    zb = np.zeros((ndif))
    B = np.concatenate((zb, B))

    # transforms C(q) and D(q) to the same polynomial base
    # order of C(q)
    m = C.shape[0] - 1
    # order of D(q)
    md = D.shape[0] - 1
    # add zeros to D(q)
    mdif = m - md
    zd = np.zeros((mdif))
    D = np.concatenate((zd, D))
    
    # creates the lag vector
    N = 2 * maxlag + 1
    tau = np.linspace(- maxlag, maxlag, N)  # linspace(start, stop, numberofpoints)
    
    # transforms B(q) and D(q) to the same polynomial base to use np.convolve()
    B = np.concatenate((B, np.zeros((m))))
    D = np.flip(D)
    D = np.concatenate((np.zeros((n)), D))
    # polynomial multiplication with np.convolve()
    H = np.convolve(B, D)
    # fixing the dimension after np.convolve()
    H = H[n : 2 * n + m + 1]

    # assembling the linear equations
    M1 = np.zeros((n + m + 1, n + m + 1))
    M2 = np.zeros((n + m + 1, n + m + 1))

    # loop that assembles the equations related to the F(q) unknown
    for k in range (0, n + 1):
        M1[k : k + m + 1, k] = np.flip(C)
    
    # loop that assembles the equations related to the G(q^-1) unknown
    for j in range (0, m):
        M2[j : j + n + 1, n + j + 1] = A.T
        
    # sum M1 and M2 to produce the full matrix of the system    
    M = M1 + M2

    # solving the linear system of equations
    x = np.linalg.solve(M, H)

    # separating the unknowns
    f = x[0 : n + 1]
    g = x[n + 1: n + m + 1]
    g = np.flip(g)
    
    # preallocating the positive and negative portion of ryw
    rywp = np.zeros((maxlag + 1))
    rywn = np.zeros((maxlag))
    
    # calculating the first coefficient of the correlation function for positive tau
    rywp[0] = f[0]
    # calculating the second coefficient of the correlation function for positive tau
    if f.shape[0] > 1:
        rywp[1] = f[1] - A[1]*rywp[0]
    
    # loop that calculates the other coefficients for positive tau
    for k in range(2, maxlag + 1):
        Sum = 0
        if k <= n:
            for j in range(1, k + 1):
                Sum = Sum + A[j] * rywp[k - j]
            rywp[k] = f[k] - Sum
        else:
            for j in range(1, n + 1):
                Sum = Sum + A[j] * rywp[k - j]
                rywp[k] = - Sum
                
    # calculates the coefficients for negative tau
    if g.shape[0] > 0:
        rywn[0] = g[0]

    # loop that calculates the other coefficients for negative tau
    for k in range(1, maxlag):
        Sum2 = 0
        if k <= m - 1:
            for j in range(1, k + 1):
                Sum2 = Sum2 + C[j] * rywn[k - j]
            rywn[k] = g[k] - Sum2
        else:
            for j in range(1, m + 1):
                Sum2 = Sum2 + C[j] * rywn[k - j]
            rywn[k] = - Sum2

    # flip the negative portion of the cross correlation function
    rywn = np.flip(rywn)
    
    # concatenate the negative and positive portions
    ryw = np.concatenate((rywn, rywp))

    # scales with the variance of e(t)
    ryw = var * ryw
    
    # returns the cross correlation function and the tau (lag) vector
    return ryw, tau