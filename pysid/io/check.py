"""
    Module for checking the arguments of several functions
"""

# Imports
from numpy import array, amax, ndarray, expand_dims, size, shape
# Variables
__all__ = ['chckin']


# functions
def chckin(na, nb, nc, nd, nf, nk, u, y):
    """
    Function used to handle input argument and throw errors
    """
    # Check if is at least a list or array
    if not isinstance(na, (int, list, ndarray)) or not isinstance(nb, (int, list, ndarray)) or\
       not isinstance(nc, (int, list, ndarray)) or not isinstance(nd, (int, list, ndarray)) or\
       not isinstance(nf, (int, list, ndarray)) or not isinstance(nk, (int, list, ndarray)) or\
       not isinstance(u, (int, list, ndarray)) or not isinstance(y, (int, list, ndarray)):
        raise Exception('Input arguments must be either list or numpy.ndarray type')
    # Verify if the arguments are lists and transform them in 2D arrays
    if isinstance(na, (int, list)):
        na = array(na, ndmin=2)
    if isinstance(nb, (int, list)):
        nb = array(nb, ndmin=2)
    if isinstance(nc, (int, list)):
        nc = array(nc, ndmin=2)
    if isinstance(nd, (int, list)):
        nd = array(nd, ndmin=2)
    if isinstance(nf, (int, list)):
        nf = array(nf, ndmin=2)
    if isinstance(nk, (int, list)):
        nk = array(nk, ndmin=2)
    if isinstance(u, (int, float, list, tuple)):
        u = array(u, ndmin=2)
    if isinstance(y, (int, float, list, tuple)):
        y = array(y, ndmin=2)
    # Check dimension
    if len(u.shape) < 2:
        u = expand_dims(u, axis=1)
    if len(y.shape) < 2:
        y = expand_dims(y, axis=1)
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


