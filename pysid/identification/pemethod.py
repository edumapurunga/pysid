# This modules provides prediction error methods for discrete-time linear
# models
"""
Module for Prediction error methods
"""

# Imports
from numpy import arange, asarray, array, append, asscalar, copy, count_nonzero, ones,\
delete, dot, empty, sum, size, amax, matrix, concatenate, shape, zeros, kron,\
eye, reshape, convolve, sqrt, where, nonzero, correlate, equal, ndarray, pi, \
absolute, exp, log, real, issubdtype, integer
from scipy.linalg import qr, solve, toeplitz
from numpy.linalg import matrix_rank
from scipy.signal import lfilter, periodogram
from scipy.optimize import leastsq, least_squares
import numpy.fft as fft

# functions
__all__ = ['fir', 'arx', 'armax', 'oe', 'bj', 'pem']

# Implementation
def fir(nb, nk, u, y):
    """
    This function estimates a FIR model based on the input-ouput data provided
    in the form of vectors (u, y). This particular function returns the
    polynomial B(q) from the following FIR model:
        y(t) = B(q)u(t) + e(t)
    Inputs:
        nb - a scalar (or a matrix ny x nu)
        nk - a scalar (or a matrix ny x nu)
        u  - data input - vector (or matrix N x nu)
        y  - data output - vector (or matrix N x ny)
    Outputs:
        B - vector containing the polynomial B(q)
    """
    # Transform everything in array for use with numpy
    _, nb, _, _, _, nk, u, y = chckin([], nb, [], [], [], nk, u, y)
    # Input handling
    Ny, ny = shape(y)
    Nu, nu = shape(u)
    nbk = nb + nk
    L = amax(nbk)
    Iny = eye(ny)
    db = sum(sum(nb+1))
    #d = da + db
    phiu = zeros(((Nu-L)*ny, db))
    k = 0;
    # Input regressors
    for i in range(0, ny):
        for j in range(0, nu):
            if (nb[i, j] > -1):
                phiu[:, k:k+nb[i, j]+1] = kron(toeplitz(u[L-nk[i, j]:Nu-nk[i, j], j], u[L-nk[i, j]-nb[i, j]:L-nk[i, j]+1, j][::-1]), Iny[:, i:i+1])
                k += nb[i, j] + 1
    # Solve the Ls problem
    phi = copy(phiu)
    y = reshape(y[L:Ny, :], ((Ny-L)*ny, 1))
    theta = qrsol(phi, y)[0]
    b = theta[0:]
    # Output
    B = empty((ny, nu), dtype='object')
    k = 0
    for i in range(0, ny):
        for j in range(0, nu):
            B[i, j] = append(zeros((1, nk[i, j])), b[k:k+nb[i, j]+1])
            k += nb[i,j] + 1
    return B

def arx(na, nb, nk, u, y, opt=0):
    """
    This function estimates a ARX model based on the input-ouput data provided
    in the form of vectors (u, y). This particular function returns the
    polynomials A(q) and B(q) from the following ARX model:
        A(q)y(t) = B(q)u(t) + e(t)
    Inputs:
        na - a scalar (or a matrix ny x ny)
        nb - a scalar (or a matrix ny x nu)
        nk - a scalar (or a matrix ny x nu)
        u  - data input - vector (or matrix N x nu)
        y  - data output - vector (or matrix N x ny)
    Outputs:
        A - vector containing the polynomial A(q)
        B - vector containing the polynomial B(q)
    """
    # Transform everything in array for use with numpy
    na, nb, _, _, _, nk, u, y = chckin(na, nb, [], [], [], nk, u, y)
    # Input handling
    Ny, ny = shape(y)
    Nu, nu = shape(u)
    nbk = nb + nk
    L = amax([amax(na), amax(nbk)])
    # MIMO case
    A = empty((ny, ny), dtype='object')
    B = empty((ny, nu), dtype='object')
    Iny = eye(ny)
    da = sum(sum(na))
    db = sum(sum(nb+1))
    #d = da + db
    phiy = zeros(((Ny-L)*ny, da))
    phiu = zeros(((Nu-L)*ny, db))
    ka = 0
    kb = 0
    # Output regressors and Input Regressors
    for i in range(0, ny):
        # Input
        for j in range(0, nu):
            if (nb[i, j] > -1):
                #phiu[:, kb:kb+nb[i, j]+1] = kron(toeplitz(u[L-nk[i, j]:-nk[i, j], j], u[L-nk[i, j]-nb[i, j]:L-nk[i, j]+1, j][::-1]), Iny[:, i:i+1]) #Should be Iny[:,j:j+1]
                phiu[:, kb:kb+nb[i, j]+1] = kron(toeplitz(u[L-nk[i, j]:Nu-nk[i, j], j], u[L-nk[i, j]-nb[i, j]:L-nk[i, j]+1, j][::-1]), Iny[:, i:i+1])
                kb += nb[i, j] + 1
        # Output
        for j in range(0, ny):
            if (na[i, j] > 0):
                phiy[:, ka:ka+na[i,j]] = kron(-toeplitz(y[L-1:-1, j], y[L-na[i, j]:L, j][::-1]),Iny[:, i:i+1])
                ka += na[i,j]
    # Solve the Ls problem
    phi = concatenate((phiy, phiu), axis=1)
    y = reshape(y[L:Ny, :], ((Ny-L)*ny, 1))
    theta = qrsol(phi, y)[0]
    a = theta[0:da]
    b = theta[da:da+db+1]
    # Prepare the results
    ka = 0
    kb = 0
    for i in range(0, ny):
        for j in range(0, nu):
            B[i, j] = append(zeros((1, nk[i, j])), b[kb:kb+nb[i,j]+1])
            kb += nb[i,j] + 1
        for j in range(0, ny):
            if (i == j):
                A[i, j] = append([1], a[ka:ka+na[i,j]])
            else:
                A[i, j] = append([0], a[ka:ka+na[i,j]])
            ka += na[i, j]
    return [A, B]

def armax(na, nb, nc, nk, u, y):
    """
    This function estimates an ARMAX model based on the input-ouput data provided
    in the form of vectors (u, y). This particular function returns the
    polynomials A(q), B(q) and C(q) from the following ARMAX model:
        A(q)y(t) = B(q)u(t) + C(q)e(t)
    Inputs:
        na - a scalar (or a matrix ny x ny)
        nb - a scalar (or a matrix ny x nu)
        nc - a scalar (or a matrix ny x 1)
        nk - a scalar (or a matrix ny x nu)
        u  - data input - vector (or matrix N x nu)
        y  - data output - vector (or matrix N x ny)
    Outputs:
        A - vector containing the polynomial A(q)
        B - vector containing the polynomial B(q)
        C - vector containing the polynomial C(q)
    """
    # Transform everything into array
    na, nb, nc, _, _, nk, u, y = chckin(na, nb, nc, [], [], nk, u, y)
    #Input Handling
    Nu, nu = shape(u)
    Ny, ny = shape(y)
    # Define the prediction error
    def pe(theta, na, nb, nc, nk, u, y):
        Ny, ny = shape(y)
        Nu, nu = shape(u)
        A = theta[0:sum(na)]
        ao = append([1], A[0:na[0]])
        #b = append(zeros((1, nk)), theta[sum(na):sum(na)+nb+1])
        c = append([1], theta[sum(na)+sum(nb+1):sum(na)+sum(nb+1)+nc+1])
        e = lfilter(ao, c, y[:,0], axis=0)
        k = sum(na)
        for i in range(0, nu):
            b = append(zeros((1, nk[i])), theta[k:k+nb[i]+1])
            k = k + nb[i] + 1
            e -= lfilter(b, c, u[:,i], axis=0)
        k = na[0]
        for i in range(1, ny):
            a = append([0], A[k:k+na[i]])
            e -= lfilter(a, c, -y[:,i], axis=0)
            k = k + na[i]
        return e
    # Initial Guess
    A = empty((ny, ny), dtype=object)
    B = empty((ny, nu), dtype=object)
    C = empty((ny,), dtype = object)
    for i in range(0, ny):
        A_ = []
        B_ = []
        E = copy(y[:,i:i+1])
        # Influence of the inputs
        for j in range(0, nu):
            a, b = ls(na[i, i], nb[i,j], nk[i,j], u[:,j:j+1], y[:,i:i+1])
            #A_ = append(A_, a)
            B_ = append(B_, b)
            E -= lfilter(append(zeros((1, nk[i,j])), b), append([1], a), u[:, j:j+1], axis=0) #u[:,i:i+1]
        A_ = append(A_, a)
        # Influence of other outputs
        for j in range(0, ny):
            if j != i:
                a, b = ls(na[i, i], na[i, j]-1, 1, y[:,j:j+1], y[:,i:i+1])
                A_ = append(A_, b)
                E -= lfilter(append([0], b), append([1], a), y[:,j:j+1], axis=0)
        # ARMA model
        _, c = arma(0, nc, E)
        # Estimate the white noise
        c = c[1:]
        #c = [0.8, -0.1]
        B_ = append(B_, c)
        thetai = append(A_, B_)
        # Solve the minimization problem
        tar = y[0:Ny,i:i+1]
        tarna = na[i, i].reshape((1,))
        index = arange(ny)
        index = delete(index, i)
        Y = concatenate((tar, y[0:Ny,index]), axis=1)
        NA = concatenate((tarna, na[i][index]))
        sol = least_squares(pe, thetai, args=(NA, nb[i], nc[i], nk[i], u.reshape(Nu, nu), Y.reshape((Ny, ny))))
        theta = sol.x
        C[i] = append([1], theta[sum(na[i])+sum(nb[i]+1):sum(na[i])+sum(nb[i]+1)+nc[i]+1])
        k = sum(na[i])
        for j in range(0, nu):
            B[i, j] = append(zeros((1, nk[i,j])), theta[k:k+nb[i, j]+1])
            k += nb[i, j] + 1
        k = 0
        for j in range(0, ny):
            if (i == j):
                A[i, j] = append([1], theta[k:k+na[i, j]])
            else:
                A[i, j] = append([0], theta[k:k+na[i, j]])
            k += na[i, j]
    return [A, B, C]

def oe(nb, nf, nk, u, y):
    """
    This function estimates an OE model based on the input-ouput data provided
    in the form of vectors (u, y). This particular function returns the
    polynomials F(q) and B(q)  from the following OE model:
        y(t) = (B(q)/F(q))u(t) + e(t)
    Inputs:
        nb - a scalar (or a matrix ny x nu)
        nf - a scalar (or a matrix ny x nu)
        nk - a scalar (or a matrix ny x nu)
        u  - data input - vector (or matrix N x nu)
        y  - data output - vector (or matrix N x ny)
    Outputs:
        B - vector containing the polynomial B(q)
        F - vector containing the polynomial F(q)
    """
    # Transform everything into array
    _, nb, _, _, nf, nk, u, y = chckin([], nb, [], [], nf, nk, u, y)
    # Input Handling
    Nu, nu = shape(u)
    Ny, ny = shape(y)
    # Define the prediction error
    def pe(theta, nf, nb, nk, u, y):
        Nu, nu = shape(u)
        F = theta[0:sum(nf)]
        B = theta[sum(nf):sum(nf)+sum(nb+1)+2]
        kf = 0
        kb = 0
        e = copy(y)
        for i in range(0, nu):
            f = append([1], F[kf:kf+nf[i]])
            b = append(zeros((1, nk[i])), B[kb:kb+nb[i]+1])
            if size(b) == 0:
                b = [0]
            kf += nf[i]
            kb += nb[i] + 1
            yu = lfilter(b, f, u[:,i], axis=0)
            e -= yu
                #e -= lfilter(b, f, u[:,i], axis=0)
                #TODO verify the strange lfilter behavior here too
        return e
    # Initialization
    B = empty((ny, nu), dtype=object)
    F = empty((ny, nu), dtype=object)
    for j in range(0, ny):
        A = []
        B_ = []
        yn = copy(y[:,j:j+1])
        for i in range(0, nu):
            a, b = ls(nf[j, i], nb[j, i], nk[j, i], u[:,i:i+1], yn)
            A = append(A, a)
            B_ = append(B_, b)
            a = append([1], a)
            b = append(zeros((1, nk[j, i])), b)
            if size(b) == 0:
                b = [0]
            yn -= lfilter(b, a, u[:,i:i+1], axis=0)
        thetai = append(A, B_)
        # Solve the minimization problem
        sol = least_squares(pe, thetai, args=(nf[j], nb[j], nk[j], u.reshape((Nu, nu)), y[:,j]))
        # Output
        theta = sol.x
        kf = 0
        kb = 0
        f = theta[0:sum(nf[j])]
        b = theta[sum(nf[j]):sum(nf[j])+sum(nb[j]+1)+2]
        for i in range(0, nu):
            B[j, i] = append(zeros((1, nk[j, i])), b[kb:kb+nb[j, i]+1])
            F[j, i] = append([1], f[kf:kf+nf[j, i]])
            kf += nf[j, i]
            kb += nb[j, i] + 1
    return [B, F]

def bj(nb, nc, nd, nf, nk, u, y):
    """
    This function estimates a BJ model based on the input-ouput data provided
    in the form of vectors (u, y). This particular function returns the
    polynomials B(q), F(q), C(q) and D(q)  from the following BJ model:
        y(t) = (B(q)/F(q))u(t) + (C(q)/D(q))e(t)
    Inputs:
        nb - a scalar (or a matrix ny x nu)
        nc - a scalar (or a matrix ny x 1)
        nd - a scalar (or a matrix ny x 1)
        nf - a scalar (or a matrix ny x nu)
        nk - a scalar (or a matrix ny x nu)
        u  - data input - vector (or matrix N x nu)
        y  - data output - vector (or matrix N x ny)
    Outputs:
        B - vector containing the polynomial B(q)
        C - vector containing the polynomial C(q)
        D - vector containing the polynomial D(q)
        F - vector containing the polynomial F(q)
    """
    _, nb, nc, nd, nf, nk, u, y = chckin([], nb, nc, nd, nf, nk, u, y)
    # Input Handling
    Nu, nu = shape(u)
    Ny, ny = shape(y)
    # Number of parameters do estimate
    #dp = nf + nb + nc + nd + 1
    # Define the prediction error
    def pe(theta, nf, nb, nc, nd, nk, u, y):
        Nu, nu = shape(u)
        F = theta[0:sum(nf)]
        B = theta[sum(nf):sum(nf)+sum(nb+1)]
        c = theta[sum(nf)+sum(nb+1):nc+sum(nf)+sum(nb+1)]
        d = theta[nc+sum(nf)+sum(nb+1):nd+nc+sum(nf)+sum(nb+1)]
        c = append([1], c)
        d = append([1], d)
        e = lfilter(d, c, y, axis=0)
        kf = 0
        kb = 0
        for i in range(0, nu):
            f = append([1], F[kf:kf+nf[i]])
            b = append(zeros((1, nk[i])), B[kb:kb+nb[i]+1])
            kf += nf[i]
            kb += nb[i] + 1
            db = convolve(b, d)
            fc = convolve(f, c)
            e -= lfilter(db, fc, u[0:,i], axis=0)
        return e
    # Initial Guess
    B = empty((ny, nu), dtype=object)
    C = empty((ny, ), dtype=object)
    D = empty((ny, ), dtype=object)
    F = empty((ny, nu), dtype=object)
    # TODO: Verify a way to compute an ARMA process
    for j in range(0, ny):
        thetaf = []
        thetab = []
        yn = copy(y[:, j:j+1])
        for i in range(0, nu):
            a, b = ls(nf[j, i], nb[j, i], nk[j, i], u[:, i:i+1], y[:,j:j+1])
            #Verify empty arrays
            if nf[j, i] == 0:
                thetab = append(thetab, b)
                yn -= lfilter(b, [1], u[:,i:i+1], axis=0)
            elif nb[j, i] == -1:
                thetai = append(thetaf, a)
                yn -= lfilter([1], a, u[:, i:i+1], axis=0)
            else:
                thetaf = append(thetaf, a)
                thetab = append(thetab, b)
                yn -= lfilter(b, a, u[:, i:i+1], axis=0)
        thetai = append(thetaf, thetab)
        ci = min(j, nu-1)
        d, c = ls(nd[j], nc[j]-1, 1, u[:, ci:ci+1], yn)
        # Verify Empty Arrays
        if nc[j] == 0:
            thetai = append(thetai, d)
        elif nd[j] == 0:
            thetai = append(thetai, c)
        else:
            thetai = append(thetai, (c, d))
        # Solve the minimization problem
        sol = least_squares(pe, thetai, args=(nf[j], nb[j], nc[j], nd[j], nk[j], u.reshape(Nu, nu), y[:,j]))
        theta = sol.x
        #B[j] = append(zeros((1, nk[j])), theta[nf[j]:nf[j]+nb[j]+1])
        C[j] = append([1], theta[sum(nf[j])+sum(nb[j]+1):nc[j]+sum(nf[j])+sum(nb[j]+1)])
        D[j] = append([1], theta[nc[j]+sum(nf[j])+sum(nb[j]+1):nd[j]+nc[j]+sum(nf[j])+sum(nb[j]+1)])
        #F[j] = append([1], theta[0:nf[j]])
        F_ = theta[0:sum(nf[j])]
        B_ = theta[sum(nf[j]):sum(nf[j])+sum(nb[j]+1)]
        kf = 0
        kb = 0
        for i in range(0, nu):
            F[j, i] = append([1], F_[kf:kf+nf[j, i]])
            B[j, i] = append(zeros((1, nk[j, i])), B_[kb:kb+nb[j, i]+1])
            kf += nf[j, i]
            kb += nb[j, i] + 1
    return [B, C, D, F]

#%% Auxiliary functions
def ls(na, nb, nk, u, y):
    '''
    :param na: number of poles from A;
    :param nb: number of zeros from B;
    :param u: input signal;
    :param y: output signal;
    :param nk: input signal delay;
    :return: coefficients of A and B in this order;
    '''
    # Asking for nothing return
    if na == 0 and nb == -1:
        return [[], []]
    # Number of samples
    Ny, ny = shape(y)
    Nu, nu = shape(u)
    # Vetor u and y must have same amount of samples
    if Ny != Nu:
        raise ValueError('Y and U must have same length!')
    # Number of coefficients to be estimated
    # (a_1, a_2, a_3,..., a_na, b_0, b_1, b_2, b_nb)
    M = na + nb + 1
    # Delay maximum needed
    L = amax([na, nb + nk], initial=0)
    # In order to estimate the coeffiecients, we will need to delay the samples.
    # If the maximum order is greater than the number of samples,
    # then it will not be possible!
    if not (Ny - L > 0):
        raise ValueError('Number of samples should be greater' &
                         'than the maximum order!')
    # Build matrix phi in which will contain y and u shifted in time
    phi = concatenate((toeplitz(-y[L-1:Ny-1], -y[L-na:L][::-1]), toeplitz(u[L-nk:Nu-nk], u[L-nk-nb:L-nk+1][::-1])), axis=1)
    # Crop y from n_max to N
    y = y[L:Ny]
    # Find theta by QR factorization
    theta = qrsol(phi, y.reshape(Ny-L,1))[0]
    # If the experiment is not informative:
    #if (matrix_rank(R) < M):
    #    raise ValueError('Experiment is not informative')
    #S = dot(phi.T, y)
    #theta = solve(R, S)
    # Split theta in vectors a and b
    a = theta[0:na]
    b = theta[na:na + nb + 1]
    return [a, b]

def qrsol(A, B):
    """
    Solve the least squares problem using QR-factorization
    """
    r, d = shape(A)
    M = concatenate((A, B), axis=1)
    Q, R = qr(M)
    R1 = R[0:d, 0:d]
    R2 = R[0:d, d]
    V = R[-1, -1]
    theta = solve(R1, R2)
    return [theta, V]

def chckin(na, nb, nc, nd, nf, nk, u, y):
    """
    Function used to handle input argument and throw errors
    """
    # Check if is at least a list or array
    if not isinstance(na, (int, list, ndarray)) or not isinstance(nb, (int, list, ndarray)) or\
       not isinstance(nc, (int, list, ndarray)) or not isinstance(nd, (int, list, ndarray)) or\
       not isinstance(nf, (int, list, ndarray)) or not isinstance(nk, (int, list, ndarray)) or\
       not isinstance(u, (int, list, ndarray)) or not isinstance(y, (int, list, ndarray)):
        raise Exception('Input arguments must be either list or array type')
    # Verify if the arguments are lists and transform them in 2D arrays
    if isinstance(na, (int, list)):
        na = array(na, ndmin=2)
    if isinstance(nb, (int, list)):
        nb = array(nb, ndmin=2)
    if isinstance(nc, (int, list)):
        nc = array(nc, ndmin=1)
    if isinstance(nd, (int, list)):
        nd = array(nd, ndmin=1)
    if isinstance(nf, (int, list)):
        nf = array(nf, ndmin=2)
    if isinstance(nk, (int, list)):
        nk = array(nk, ndmin=2)
    if isinstance(nb, (int, float, list)):
        u = array(u, ndmin=2)
    if isinstance(nb, (int, float, list)):
        y = array(y, ndmin=2)
    # Check the shapes
    Ny, ny = shape(y)
    Nu, nu = shape(u)
    ra, ca = shape(na)
    rb, cb = shape(nb)
    rc = shape(nc)[0]
    rd = shape(nd)[0]
    rf, cf = shape(nf)
    rk, ck = shape(nk)
    L = int(amax([amax(na, initial=0), amax(nb + nk, initial=0), amax(nc, initial=0), amax(nd, initial=0), amax(nf, initial=0)]))
    # Different number of Data
    if Ny != Nu:
        raise Exception('Input and Output must be the same number of data samples')
    if Ny < L:
        raise Exception('Not enough data for model identification')
    # Classify Into the structures: Initial Variables
    #isAR = True
    #isARX = True
    #isARMA = True
    #isARMAX = True
    #isBB = True
    #isBJ = True
    #isFIR = True
    # Verify the orders' shapes
    if (size(na) != 0) and (ra != ca or ra != ny):
        raise Exception('na must have shape (ny x ny)')
    if (size(nb) != 0) and (rb != ny or cb != nu):
        raise Exception('nb must have shape (ny x nu)')
    if (size(nc) != 0) and (rc != ny):
        raise Exception('nc must have shape (ny)')
    if (size(nd) != 0) and (rd != ny):
        raise Exception('nd must have shape (ny)')
    if (size(nf) != 0) and (rf != ny or cf != nu):
        raise Exception('nf must have shape (ny x nu)')
    return [na, nb, nc, nd, nf, nk, u, y]

#%% Testing functions
def pem(A, B, C, D, F, u, y, mu=[] ,solver='lm'):
    """
    This functions implements the prediction error method for the gerenal
    tranfer function black-box model:
        A(q)y(t) = [B(q)/F(q)]u(t) + [C(q)/D(q)]e(t)
    Inputs:
        A - A (ny x ny) numpy object filled with polynomials
        B - A (ny x nu) numpy object filled with polynomials
        C - A (ny x 1) numpy object filled with polynomials
        D - A (ny x 1) numpy object filled with polynomials
        F - A (ny x nu) numpy object filled with polynomials
        u - The system input (N x nu)
        y - The system otput (N x ny)
        mu- A mask representing the unknowns
    Outputs:
        A
        B
        C
        D
        E
        F
    """
    #Array everything
    A = array(A, ndmin=2, dtype='object')
    #B = array(B, ndmin=2, dtype='object')
    C = array(C, ndmin=1, dtype='object')
    D = array(D, ndmin=1, dtype='object')
    #F = array(F, ndmin=2, dtype='object')
    y = array(y)
    u = array(u)
    #Input Handling
    Ny, ny = shape(y)
    Nu, nu = shape(u)
    #Empty mask
    mu = array(mu)
    if size(mu) == 0:
        kn = False
    else:
        muA = array(mu[0])
        muB = array(mu[1])
        muC = array(mu[2])
        muD = array(mu[3])
        muF = array(mu[4])
        kn = True
    #Error Handling
    if Ny != Nu:
        raise ValueError('The data must have the same number of samples')
    #Configure the output
    Aout = empty((ny, ny), dtype='object')
    Bout = empty((ny, nu), dtype='object')
    Cout = empty((ny, ), dtype='object')
    Dout = empty((ny, ), dtype='object')
    Fout = empty((ny, nu), dtype='object')
    #SISO case
    if nu == 1 and ny == 1:
        #Recover polynomial orders
        na = size(A)-1
        nk = 0
        fc = True
        for i in range(0, size(B)):
            if (B[i] == 0 and fc):
                nk += 1
            else:
                fc = False
                nb = size(B) - nk - 1
                break
        nc = size(C)-1
        nd = size(D)-1
        nf = size(F)-1
        #Define the prediction error
        def pe(theta, A, B, C, D, F, u, y, mu, kn):
            #Take the orders
            na = size(A)-1
            nk = 0
            fc = True
            for i in range(0, size(B)):
                if (B[i] == 0 and fc):
                    nk += 1
                else:
                    fc = False
                    nb = size(B) - nk - 1
                    break
            nc = size(C)-1
            nd = size(D)-1
            nf = size(F)-1
            #Verify what parameters are constant
            if kn:
                #Known Elements
                muA = array(mu[0])
                muB = array(mu[1])
                muC = array(mu[2])
                muD = array(mu[3])
                muF = array(mu[4])
                #Number of known elements
                kA = count_nonzero(muA)
                kB = count_nonzero(muB)
                kC = count_nonzero(muC)
                kD = count_nonzero(muD)
                kF = count_nonzero(muF)
                #Number of unknown elements
                ukA = na - kA
                ukB = nb - kB + 1
                ukC = nc - kC
                ukD = nd - kD
                ukF = nf - kF
                #Copy arrays
                a = copy(A)
                b = copy(B)
                c = copy(C)
                d = copy(D)
                f = copy(F)
                #Fill it with the unknown elements
                k = 0
                if ukA != 0:
                    i = array(where(muA==0))[0]
                    i = i[1:]
                    a[i] = theta[k:k+ukA]
                    k += ukA
                if ukB != 0:
                    i = array(where(muB==0))[0]
                    i = i[nk:]
                    b[i] = theta[k:k+ukB]
                    k += ukB
                if ukC != 0:
                    i = array(where(muC==0))[0]
                    i = i[1:]
                    c[i] = theta[k:k+ukC]
                    k += ukC
                if ukD != 0:
                    i = array(where(muD==0))[0]
                    i = i[1:]
                    d[i] = theta[k:k+ukD]
                    k+= ukD
                if ukF != 0:
                    i = array(where(muF==0))[0]
                    i = i[1:]
                    f[i] = theta[k:k+ukF]
            #Everything is unknown
            else:
                k = 0
                if na != 0:
                    a = append([1], theta[k:k+na])
                    k += na
                else:
                    a = 1
                if nb != -1:
                    b = append(zeros((1, nk)), theta[k:k+nb+1])
                    k += nb + 1
                else:
                    b = 0
                if nc != 0:
                    c = append([1], theta[k:k+nc])
                    k += nc
                else:
                    c = 1
                if nd != 0:
                    d = append([1], theta[k:k+nd])
                    k += nd
                else:
                    d = 1
                if nf != 0:
                    f = append([1], theta[k:k+nf])
                else:
                    f = 1
                #e = (AD)/(C)(y - (B)/(AF)u)
            ad = convolve(a, d)
            bd = convolve(b, d)
            cf = convolve(c, f)
            e = lfilter(ad, c, y, axis=0) - lfilter(bd, cf, u, axis=0)
            return e
        #Initial Guess
        #a, b = ls(na+nf, nb, nk, u, y)
        #yn = y - lfilter(append(zeros((1,nk)), b), append([1], a), u, axis=0)
        #f = a[na:na+nf+1]
        #a = a[0:na]
        #d, c = ls(nd, nc-1, 1, u, yn)
        #Verify what parameters are previously known
        if kn:
            #Copy arrays
            pa = A[1:na+1]
            pb = B[nk:nk+nb+1]
            pc = C[1:nc+1]
            pd = D[1:nd+1]
            pf = F[1:nf+1]
            #Fill it with the unknown elements
            i = where(muA[1:na+1]==0)
            a = pa[i]
            i = where(muB[nk:nk+nb+1]==0)
            b = pb[i]
            i = where(muC[1:nc+1]==0)
            c = pc[i]
            i = where(muD[1:nd+1]==0)
            d = pd[i]
            i = where(muF[1:nf+1]==0)
            f = pf[i]
        else:
            a = A[1:na+1]
            b = B[nk:nk+nb+1]
            c = C[1:nc+1]
            d = D[1:nd+1]
            f = F[1:nf+1]
        thetai = concatenate((a, b, c, d, f))
        #Call minimization function
        sol = least_squares(pe, thetai, args=(A, B, C, D, F, u.reshape(Nu,), y.reshape(Ny,), mu, kn))
        theta = sol.x
        k = 0
        if kn:
            #Number of known elements
            kA = count_nonzero(muA)
            kB = count_nonzero(muB)
            kC = count_nonzero(muC)
            kD = count_nonzero(muD)
            kF = count_nonzero(muF)
            #Number of unknown elements
            ukA = na - kA
            ukB = nb - kB + 1
            ukC = nc - kC
            ukD = nd - kD
            ukF = nf - kF
            #Fill it with the unknown elements
            k = 0
            if ukA != 0:
                i = array(where(muA==0))[0]
                i = i[1:]
                A[i] = theta[k:k+ukA]
                k += ukA
            if ukB != 0:
                i = array(where(muB==0))[0]
                i = i[nk:]
                B[i] = theta[k:k+ukB]
                k += ukB
            if ukC != 0:
                i = array(where(muC==0))[0]
                i = i[1:]
                C[i] = theta[k:k+ukC]
                k += ukC
            if ukD != 0:
                i = array(where(muD==0))[0]
                i = i[1:]
                D[i] = theta[k:k+ukD]
                k+= ukD
            if ukF != 0:
                i = array(where(muF==0))[0]
                i = i[1:]
                F[i] = theta[k:k+ukF]
        else:
            if na != 0:
                A = append([1], theta[k:k+na])
                k += na
            else:
                A = 1
            if nb != -1:
                B = append(zeros((1, nk)), theta[k:k+nb+1])
                k += nb + 1
            else:
                B = 0
            if nc != 0:
                C = append([1], theta[k:k+nc])
                k += nc
            else:
                C = 1
            if nd != 0:
                D = append([1], theta[k:k+nd])
                k += nd
            else:
                D = 1
            if nf != 0:
                F = append([1], theta[k:k+nf])
            else:
                F = 1
    #MISO case
    elif (ny==1):
        #Define the output of the algorithm
        #B = empty((nu,), dtype='object')
        #F = empty((nu,), dtype='object')
        nf = zeros((nu,), dtype = 'int32')
        nb = zeros((nu,), dtype = 'int32')
        nk = zeros((nu,), dtype = 'int32')
        #Recover the polynomial orders
        for i in range(0, nu):
            nf[i] = size(F[0, i])-1
            fc = True
            for j in range(0, size(B[0, i])): #B[i]
                if (B[0, i][j] == 0 and fc):
                    nk[i] += 1
                else:
                    fc = False
                    nb[i] = size(B[0, i]) - nk[i] - 1
                    break
        na = size(A)-1
        nc = size(C)-1
        nd = size(D)-1
        #Define the prediction error
        #Define the prediction error
        def pe(theta, A, B, C, D, F, u, y, mu, kn):
            #Check shape of u
            Nu, nu = shape(u)
            #Take the orders
            nf = zeros((nu,), dtype = 'int32')
            nb = zeros((nu,), dtype = 'int32')
            nk = zeros((nu,), dtype = 'int32')
            for i in range(0, nu):
                fc = True
                nf[i] = size(F[0, i]) - 1
                for j in range(0, size(B[0, i])):
                    if (B[0, i][j] == 0 and fc):
                        nk[i] += 1
                    else:
                        fc = False
                        nb[i] = size(B[0, i]) - nk[i] - 1
                        break
            na = size(A)-1
            nc = size(C)-1
            nd = size(D)-1
            #Verify what parameters are constant
            if kn:
                #Known Elements
                muA = array(mu[0])
                muB = array(mu[1])
                muC = array(mu[2])
                muD = array(mu[3])
                muF = array(mu[4])
                #Number of known elements
                kA = count_nonzero(muA)
                kB = zeros((nu,), dtype = 'int32')
                kF = zeros((nu,), dtype = 'int32')
                #Unknown
                ukB = zeros((nu,), dtype = 'int32')
                ukF = zeros((nu,), dtype = 'int32')
                for i in range(0, nu):
                    #Known elements
                    kB[i] = count_nonzero(muB[i])
                    kF[i] = count_nonzero(muF[i])
                    #Unknown elements
                    ukB[i] = nb[i] - kB[i] + 1
                    ukF[i] = nf[i] - kF[i]
                kB = count_nonzero(muB)
                kC = count_nonzero(muC)
                kD = count_nonzero(muD)
                #Number of unknown elements
                ukA = na - kA
                ukC = nc - kC
                ukD = nd - kD
                #Copy arrays
                a = copy(A)
                b = copy(B)
                c = copy(C)
                d = copy(D)
                f = copy(F)
                #Fill it with the unknown elements
                k = 0
                if ukA != 0:
                    i = array(where(muA==0))[0]
                    i = i[1:]
                    a[i] = theta[k:k+ukA]
                    k += ukA
                for j in range(0, nu):
                    if ukB[j] != 0:
                        i = array(where(muB[j]==0))[0]
                        i = i[nk[j]:]
                        b[0, j][i] = theta[k:k+ukB[j]]
                        k += ukB[j]
                if ukC != 0:
                    i = array(where(muC==0))[0]
                    i = i[1:]
                    c[i] = theta[k:k+ukC]
                    k += ukC
                if ukD != 0:
                    i = array(where(muD==0))[0]
                    i = i[1:]
                    d[i] = theta[k:k+ukD]
                    k+= ukD
                for j in range(0, nu):
                    if ukF[j] != 0:
                        i = array(where(muF[j]==0))[0]
                        i = i[1:]
                        #TODO: Cannot Broadcast shape 0 into shape 1 (deal with scalar)
                        f[0, j][i] = theta[k:k+ukF[j]]
                        k += ukF[j]
            #Everything is unknown
            else:
                #Initialization
                f = empty((nu,), dtype='object')
                b = empty((nu,), dtype='object')
                k = 0
                if na != 0:
                    a = append([1], theta[k:k+na])
                    k += na
                else:
                    a = 1
                for i in range(0, nu):
                    if nb[i] != -1:
                        b[i] = append(zeros((1, nk[i])), theta[k:k+nb[i]+1])
                        k += nb[i] + 1
                    else:
                        b[i] = 0
                if nc != 0:
                    c = append([1], theta[k:k+nc])
                    k += nc
                else:
                    c = 1
                if nd != 0:
                    d = append([1], theta[k:k+nd])
                    k += nd
                else:
                    d = 1
                for i in range(0, nu):
                    if nf[i] != 0:
                        f[i] = append([1], theta[k:k+nf[i]])
                        k += nf[i]
                    else:
                        f[i] = 1
                #e = (AD)/(C)(y - (B)/(AF)u)
            ad = convolve(a[0], d[0])
            e = lfilter(ad, c[0], y, axis=0)
            for i in range(0, nu):
                bd = convolve(b[0, i], d)
                cf = convolve(c, f[0, i])
                e -= lfilter(bd, cf, u[:, i], axis=0)
            return e.tolist()
        #Verify what parameters are previously known
        if kn:
            pb = empty((sum(nb+1),), dtype = 'object')
            pf = empty((sum(nf)  ,), dtype = 'object')
            #Copy arrays
            pa = A[1:na+1]
            for i in range(0, nu):
                pb[i] = B[0, i][nk[i]:nk[i]+nb[i]+1]
                pf[i] = F[0, i][1:nf[i]+1]
            pc = C[1:nc+1]
            pd = D[1:nd+1]
            #Fill it with the unknown elements
            i = where(muA[1:na+1]==0)
            a = pa[i]
            b = []
            f = []
            for j in range(0, nu):
                i = where(equal(muB[j][nk[j]:nk[j]+nb[j]+1],0))
                b = append(b, pb[j][i])
                i = where(muF[j][1:nf[j]+1]==0)
                f = append(f, pf[j][i])
            i = where(muC[1:nc+1]==0)
            c = pc[i]
            i = where(muD[1:nd+1]==0)
            d = pd[i]
        else:
            a = A[1:na+1]
            b = []
            f = []
            for i in range(0, nu):
                b = append(b, B[0, i][nk[i]:nk[i]+nb[i]+1])
                f = append(f, F[0, i][1:nf[i]+1])
            c = C[1:nc+1]
            d = D[1:nd+1]
        #Initial Guess
        thetai = concatenate((a.tolist(), b.tolist(), c.tolist(), d.tolist(), f.tolist()))
        #Call minimization function
        sol = least_squares(pe, thetai, args=(A, B, C, D, F, u.reshape(Nu, nu), y.reshape(Ny,), mu, kn))
        theta = sol.x
        k = 0
        if kn:
            #Number of known elements
            kA = count_nonzero(muA)
            kB = zeros((nu,), dtype = 'int32')
            kF = zeros((nu,), dtype = 'int32')
            ukB = zeros((nu,), dtype = 'int32')
            ukF = zeros((nu,), dtype = 'int32')
            #Unknown
            for i in range(0, nu):
                kB[i] = count_nonzero(muB[i])
                kF[i] = count_nonzero(muF[i])
                ukB[i] = nb[i] - kB[i] + 1
                ukF[i] = nf[i] - kF[i]
            kC = count_nonzero(muC)
            kD = count_nonzero(muD)
            #Number of unknown elements
            ukA = na - kA
            ukC = nc - kC
            ukD = nd - kD
            #Fill it with the unknown elements
            k = 0
            if ukA != 0:
                i = array(where(muA==0))[0]
                i = i[1:]
                A[i] = theta[k:k+ukA]
                k += ukA
            for j in range(0, nu):
                if ukB[j] != 0:
                    i = array(where(muB[j]==0))[0]
                    i = i[nk[j]:]
                    B[0, j][i] = theta[k:k+ukB[j]]
                    k += ukB[j]
            if ukC != 0:
                i = array(where(muC==0))[0]
                i = i[1:]
                C[i] = theta[k:k+ukC]
                k += ukC
            if ukD != 0:
                i = array(where(muD==0))[0]
                i = i[1:]
                D[i] = theta[k:k+ukD]
                k+= ukD
            for j in range(0, nu):
                if ukF[j] != 0:
                    i = array(where(muF[j]==0))[0]
                    i = i[1:]
                    F[0, j][i] = theta[k:k+ukF[j]]
                    k += ukF[j]
        else:
            if na != 0:
                A = append([1], theta[k:k+na])
                k += na
            else:
                A = 1
            for i in range(0, nu):
                if nb[i] != -1:
                    B[i] = append(zeros((1, nk[i])), theta[k:k+nb[i]+1])
                    k += nb[i] + 1
                else:
                    B[i] = 0
            if nc != 0:
                C = append([1], theta[k:k+nc])
                k += nc
            else:
                C = 1
            if nd != 0:
                D = append([1], theta[k:k+nd])
                k += nd
            else:
                D = 1
            for i in range(0, nu):
                if nf[i] != 0:
                    F[i] = append([1], theta[k:k+nf[i]])
                    k += nf[i]
                else:
                    F[i] = 1
    #SIMO case
    elif (nu==1):
        do = nothing
    #MIMO case
    else:
        #Define the orders of the polynomials
        na = zeros((ny, ny), dtype = 'int32')
        nb = zeros((ny, nu), dtype = 'int32')
        nc = zeros((ny,), dtype = 'int32')
        nd = zeros((ny,), dtype = 'int32')
        nf = zeros((ny, nu), dtype = 'int32')
        nk = zeros((ny, nu), dtype = 'int32')
        #Recover the polynomial orders
        for j in range(0, ny):
            #Noise
            nc[j] = size(C[j]) - 1
            nd[j] = size(D[j]) - 1
            #Output
            for i in range(0, ny):
                na[j, i] = size(A[j, i]) - 1
            #Input
            for i in range(0, nu):
                nf[j, i] = size(F[j, i]) - 1
                fc = True
                for k in range(0, size(B[j, i])):
                    if (B[j, i][k] == 0 and fc):
                        nk[j, i] += 1
                    else:
                        fc = False
                        nb[j, i] = size(B[j, i]) - nk[j, i] - 1
                        break
        #Define the prediction error
        def pe(theta, A, B, C, D, F, u, y, mu, kn):
            #Check shape of u ans y
            Nu, nu = shape(u)
            Ny, ny = shape(y)
            #NA = concatenate((tarna, na[i][index]))
            #Take the orders
            na = zeros((ny,), dtype = 'int32')
            nf = zeros((nu,), dtype = 'int32')
            nb = zeros((nu,), dtype = 'int32')
            nk = zeros((nu,), dtype = 'int32')
            for i in range(0, ny):
                na[i] = size(A[i])-1
            for i in range(0, nu):
                fc = True
                nf[i] = size(F[i]) - 1
                for j in range(0, size(B[i])):
                    if (B[i][j] == 0 and fc):
                        nk[i] += 1
                    else:
                        fc = False
                        nb[i] = size(B[i]) - nk[i] - 1
                        break
            nc = size(C)-1
            nd = size(D)-1
            #Verify what parameters are constant
            if kn:
                #Known Elements
                muA = array(mu[0])
                muB = array(mu[1])
                muC = array(mu[2])
                muD = array(mu[3])
                muF = array(mu[4])
                #Number of known elements
                kA = zeros((ny,), dtype = 'int32')
                ukA = zeros((ny,), dtype = 'int32')
                for i in range(0, ny):
                    #Known Elements
                    kA[i] = count_nonzero(muA[i])
                    #Unknown elements
                    uKA[i] = na[i] - kA[i]
                kB = zeros((nu,), dtype = 'int32')
                kF = zeros((nu,), dtype = 'int32')
                #Unknown
                ukB = zeros((nu,), dtype = 'int32')
                ukF = zeros((nu,), dtype = 'int32')
                for i in range(0, nu):
                    #Known elements
                    kB[i] = count_nonzero(muB[i])
                    kF[i] = count_nonzero(muF[i])
                    #Unknown elements
                    ukB[i] = nb[i] - kB[i] + 1
                    ukF[i] = nf[i] - kF[i]
                kB = count_nonzero(muB)
                kC = count_nonzero(muC)
                kD = count_nonzero(muD)
                #Number of unknown elements
                ukC = nc - kC
                ukD = nd - kD
                #Copy arrays
                a = copy(A)
                b = copy(B)
                c = copy(C)
                d = copy(D)
                f = copy(F)
                #Fill it with the unknown elements
                k = 0
                for j in range(0, nu):
                    if ukA[j] != 0:
                        i = array(where(muA[j]==0))[0]
                        i = i[1:]
                        a[j][i] = theta[k:k+ukA[j]]
                        k += ukA[j]
                for j in range(0, nu):
                    if ukB[j] != 0:
                        i = array(where(muB[j]==0))[0]
                        i = i[nk[j]:]
                        b[j][i] = theta[k:k+ukB[j]]
                        k += ukB[j]
                if ukC != 0:
                    i = array(where(muC==0))[0]
                    i = i[1:]
                    c[i] = theta[k:k+ukC]
                    k += ukC
                if ukD != 0:
                    i = array(where(muD==0))[0]
                    i = i[1:]
                    d[i] = theta[k:k+ukD]
                    k+= ukD
                for j in range(0, nu):
                    if ukF[j] != 0:
                        i = array(where(muF[j]==0))[0]
                        i = i[1:]
                        f[j][i] = theta[k:k+ukF[j]]
                        k += ukF[j]
            #Everything is unknown
            else:
                #Initialization
                a = empty((ny,), dtype='object')
                f = empty((nu,), dtype='object')
                b = empty((nu,), dtype='object')
                k = 0
                for i in range(0, ny):
                    if i == 0:
                        aux = [1]
                    else:
                        aux = [0]
                    if na[i] != 0:
                        a[i] = append(aux, theta[k:k+na[i]])
                        k += na[i]
                    else:
                        a[i] = aux
                for i in range(0, nu):
                    if nb[i] != -1:
                        b[i] = append(zeros((1, nk[i])), theta[k:k+nb[i]+1])
                        k += nb[i] + 1
                    else:
                        b[i] = 0
                if nc != 0:
                    c = append([1], theta[k:k+nc])
                    k += nc
                else:
                    c = 1
                if nd != 0:
                    d = append([1], theta[k:k+nd])
                    k += nd
                else:
                    d = 1
                for i in range(0, nu):
                    if nf[i] != 0:
                        f[i] = append([1], theta[k:k+nf[i]])
                        k += nf[i]
                    else:
                        f[i] = 1
                #e = (AD)/(C)(y - (B)/(AF)u)
            ad = convolve(a[0], d)
            e = lfilter(ad, c, y[:, 0], axis=0)
            for i in range(0, nu):
                bd = convolve(b[i], d)
                cf = convolve(c, f[i])
                e -= lfilter(bd, cf, u[:, i], axis=0)
            #Additional Outputs
            for i in range(1, ny):
                ad = convolve(a[i], d)
                e -= lfilter(ad, c, -y[:,i], axis=0)
            return e
        #Solve ny MISO problems
        for l in range(0, ny):
            #Sort to make the predicted y to be the first
            #Verify what parameters are previously known
            if kn:
                pa = empty((ny,), dtype = 'object')
                pb = empty((nu,), dtype = 'object')
                pf = empty((nu,), dtype = 'object')
                #Copy arrays
                a = []
                for i in range(0, ny):
                    pa[i] = A[l, i][1:]
                    ind = where(muA[l, i][1:na[l, i]+1]==0)
                    a = append(a, pa[i][ind])
                for i in range(0, nu):
                    pb[i] = B[l, i][nk[l, i]:nk[l, i]+nb[l, i]+1]
                    pf[i] = F[l, i][1:nf[l, i]+1]
                pc = C[l][1:nc[l]+1]
                pd = D[l][1:nd[l]+1]
                #Fill it with the unknown elements
                b = []
                f = []
                for j in range(0, nu):
                    i = where(muB[j][nk[j]:nk[j]+nb[j]+1]==0)
                    b = append(b, pb[j][i])
                    i = where(muF[j][1:nf[j]+1]==0)
                    f = append(f, pf[j][i])
                i = where(muC[l][1:nc[l]+1]==0)
                c = pc[i]
                i = where(muD[l][1:nd[l]+1]==0)
                d = pd[i]
            else:
                a = empty((ny,), dtype='object')
                for i in range(0, ny):
                    a[i] = A[l, i][1:]
                b = []
                f = []
                for i in range(0, nu):
                    b = append(b, B[l, i][nk[l, i]:nk[l, i]+nb[l, i]+1])
                    f = append(f, F[l, i][1:nf[l, i]+1])
                c = C[l][1:nc[l]+1]
                d = D[l][1:nd[l]+1]
            #Organize the target
            tar = y[0:Ny,l:l+1]
            ax = a[l]
            Ax = A[l, l].reshape((1, na[l, l]+1))
            index = arange(ny)
            index = delete(index, l)
            Y = concatenate((tar, y[0:Ny,index]), axis=1)
            ax = array([ax, a[index]])
            Ax = array([Ax, A[l][index]])
            a = []
            for i in range(0, ny):
                a = append(a, ax[i])
            #Initial Guess
            thetai = concatenate((a, b, c, d, f))
            #Call minimization function
            sol = least_squares(pe, thetai, args=(Ax, B[l], C[l], D[l], F[l], u.reshape(Nu, nu), Y.reshape(Ny, ny), mu, kn))
            theta = sol.x
            k = 0
            if kn:
                #Number of known elements
                kA = count_nonzero(muA)
                kB = zeros((nu,), dtype = 'int32')
                kF = zeros((nu,), dtype = 'int32')
                ukB = zeros((nu,), dtype = 'int32')
                ukF = zeros((nu,), dtype = 'int32')
                #Unknown
                for i in range(0, nu):
                    kB[i] = count_nonzero(muB[i])
                    kF[i] = count_nonzero(muF[i])
                    ukB[i] = nb[i] - kB[i] + 1
                    ukF[i] = nf[i] - kF[i]
                kC = count_nonzero(muC)
                kD = count_nonzero(muD)
                #Number of unknown elements
                ukA = na - kA
                ukC = nc - kC
                ukD = nd - kD
                #Fill it with the unknown elements
                k = 0
                if ukA != 0:
                    i = array(where(muA==0))[0]
                    i = i[1:]
                    A[i] = theta[k:k+ukA]
                    k += ukA
                for j in range(0, nu):
                    if ukB[j] != 0:
                        i = array(where(muB[j]==0))[0]
                        i = i[nk[j]:]
                        B[j][i] = theta[k:k+ukB[j]]
                        k += ukB[j]
                if ukC != 0:
                    i = array(where(muC==0))[0]
                    i = i[1:]
                    C[i] = theta[k:k+ukC]
                    k += ukC
                if ukD != 0:
                    i = array(where(muD==0))[0]
                    i = i[1:]
                    D[i] = theta[k:k+ukD]
                    k+= ukD
                for j in range(0, nu):
                    if ukF[j] != 0:
                        i = array(where(muF[j]==0))[0]
                        i = i[1:]
                        F[j][i] = theta[k:k+ukF[j]]
                        k += ukF[j]
            else:
                #A polynomial
                for i in range(0, ny):
                    if l == i:
                        aux = [1]
                    else:
                        aux = [0]
                    if na[l, i] != 0:
                        Aout[l, i] = append(aux, theta[k:k+na[l, i]])
                        k += na[l, i]
                    else:
                        Aout[l, i] = aux
                #B polynomial
                for i in range(0, nu):
                    if nb[l, i] != -1:
                        Bout[l, i] = append(zeros((1, nk[l, i])), theta[k:k+nb[l, i]+1])
                        k += nb[l, i] + 1
                    else:
                        Bout[l, i] = 0
                #C polynomial
                if nc[l] != 0:
                    Cout[l] = append([1], theta[k:k+nc[l]])
                    k += nc[l]
                else:
                    Cout[l] = 1
                if nd[l] != 0:
                    Dout[l] = append([1], theta[k:k+nd[l]])
                    k += nd[l]
                else:
                    Dout[l] = 1
                for i in range(0, nu):
                    if nf[l, i] != 0:
                        Fout[l, i] = append([1], theta[k:k+nf[l, i]])
                        k += nf[l, i]
                    else:
                        Fout[l, i] = 1
        A = copy(Aout)
        B = copy(Bout)
        C = copy(Cout)
        D = copy(Dout)
        F = copy(Fout)
    #Find the solution of the minimzation problem
    return [A, B, C, D, F]
