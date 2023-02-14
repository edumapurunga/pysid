"""
    Solvers for the identification modules.
"""

from numpy import append, array, amax, concatenate, dot, shape, empty, dot, zeros
from scipy.linalg import qr, solve, toeplitz

# Variables
__all__ = ['ls', 'qrsol', 'burg', 'levinson']

# functions
def ls(na, nb, nk, u, y):
    ''' Least Squares solution
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
    V = R[d, d]
    theta = solve(R1, R2)
    return [theta, V, R1]

def qrsolm(psi,y):
    """
    Solve the least saqures problem using QR-factorization but for mimo systems
    """
    ny = y.shape[1];
    theta = zeros((psi.shape[1],ny))
    for i in range(ny):
        theta_i = qrsol(psi, y[:,i].reshape((y.shape[0],1)))[0]
        theta[:,i] = theta_i
    return theta
 
def levinson(R, n):
    """
    This function implements the Levinson algorithm for fast parameters computations
    """
    A = empty((n,), dtype='object')
    alfa = append(R[1], zeros((1, n)))
    E = append(R[0], zeros((1, n)))
    # Levinson Algorithm
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
    """Returns the output of the burg algorithm."""
    # Array Everything
    y = array(y)
    n = array(n)
    # Size
    N, ny = shape(y)
    # Initialization
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
