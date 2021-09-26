# Testing modules for pemethod.py using pytest
# @author: lima84
# Docstring?

import pytest
from numpy import ndarray
from numpy.random import rand, randn
from scipy.signal import lfilter

# ------------------- PYTEST -------------------
#   Running test modules with Pytest:
# 'pytest': runs all tests in the current folder
# 'pytest test_method.py': runs all tests in 'test_method.py'
# 'pytest test_method.py::test_sth': runs 'test_sth' in 'test_method.py'
# 'pytest -k "arx"': runs tests whose names contain "arx"
# 'pytest -k "not arx"': runs tests whose names does not contain "arx"
# 'pytest -m slow': runs all tests market with @pytest.mark.slow

#   Useful commands and arguments:
# '--maxfail = K': stops testing after K fails
# '-v': shows a verbose output
# '--fixtures': lists built-in and user fixtures

# ------------------- FIXTURES -------------------

# Defines a set of input and output data as a fixture
@pytest.fixture
def test_signals():
    # True test system parameters
    Ao = [1, -1.2, 0.36]
    Bo = [0, 0.5, 0.1]

    # Replicates the following experiment:
    # y(t) = Go(q)*u(t) + Ho(q)*e(t),
    # where u(t) is the system input and e(t) white noise
    N = 100                 # Number of samples
    u = -1 + 2*rand(N, 1)   # Defines input signal
    e = 0.01*randn(N, 1)    # Emulates gaussian white noise with std = 0.01

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = lfilter(Bo, Ao, u, axis=0) + lfilter([1], Ao, e, axis=0)

    return u, y

# Defines sets of arguments for tests that request the respective fixtures
# Here, the values of na, nb and nk are varied
@pytest.mark.parametrize("na, nb, nk", [(1, 0 , 1), ('1', 2, 3)])

# ------------------- TEST FUNCTIONS -------------------

# Validates ARX model orders
# Every passed argument is a requested @pytest.fixture
def test_arx_orders(na, nb, nk):
    # Checks the consistency of na 
    assert isinstance(na, int)   
    
    # Checks the consistency of nb
    assert isinstance(nb, int)   
    
    # Checks the consistency of nk
    assert isinstance(nk, int) 

def test_data_type(test_signals):
    u = test_signals[0]
    y = test_signals[1]

    # Checks the consistency of u(t)
    assert isinstance(u, ndarray) or isinstance(u, list)

    # Checks the consistency of y(t)
    assert isinstance(y, ndarray) or isinstance(y, list)