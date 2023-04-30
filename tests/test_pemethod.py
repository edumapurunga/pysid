# @author: lima84
"""
    Testing modules for pemethod.py using pytest
"""
import pytest
from numpy import array, ndarray, convolve, concatenate, zeros, dot
from numpy.random import rand, randn
from numpy.linalg import inv
from scipy.signal import lfilter
from pysid.identification.pemethod import arx,armax,bj
from pysid.identification.recursive import rls 
from scipy.stats import chi2

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

# -------------------  -------------------

# Defines a set of input and output data as a fixture

def get_value_elipse(t, t0, P):
    return (t - t0).T @ P @ (t - t0)

def check_inside_elipse(chivalue, df, alfa=0.995):
    return chivalue < chi2.ppf(alfa, df=df)

# -----------------Arx-----------------

@pytest.fixture
def test_polynomials_arx_siso():
    Ao = [1, -1.2, 0.36]
    Bo = [0, 0.5, 0.1]

    return [Ao, Bo]

@pytest.fixture #Gera u,y,Ao e Bo, o test só usará isso, mais facíl de mudar o test
def test_signals_arx_siso(test_polynomials_arx_siso):
    # True test system parameters
    Ao = test_polynomials_arx_siso[0]
    Bo = test_polynomials_arx_siso[1]

    # Replicates the following experiment:
    # y(t) = Go(q)*u(t) + Ho(q)*e(t),
    # where u(t) is the system input and e(t) white noise
    N = 100                 # Number of samples
    u = -1 + 2*rand(N, 1)   # Defines input signal
    e = 0.01*randn(N, 1)    # Emulates gaussian white noise with std = 0.01

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = lfilter(Bo, Ao, u, axis=0) + lfilter([1], Ao, e, axis=0)

    return [u, y]


def test_arx_siso(test_signals_arx_siso, test_polynomials_arx_siso):
    A = array(test_polynomials_arx_siso[0])
    B = array(test_polynomials_arx_siso[1])
    t0 = array(A[1:].tolist() + B[1:].tolist())

    u = test_signals_arx_siso[0]
    # e = 0.01*randn(1000, 1)
    y = test_signals_arx_siso[1]
    # y = lfilter(B, A, u, axis=0) + lfilter([1], A, e, axis=0)
    m = arx(2, 1, 1, u, y)
    t = m.parameters
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

# -----------------Armax-----------------
#SISO
@pytest.fixture
def test_polynomials_armax_siso(): #isso aqui define A e B para os testes q receberem test_polynomials
    Ao = [1, -1.2, 0.36]
    Bo = [0, 0.5, 0.1]
    Co = [1.0, 0.9, -0.1]
    return [Ao,Bo,Co]

@pytest.fixture
def test_signals_armax_siso(test_polynomials_armax_siso): # isso aqui define u,y, Ao e Bo para os testes que receberem test_signals
    # makes siso signals with S (ARX: G(q) = B(q)/A(q) and H(q) = C/A(q))
    Ao = test_polynomials_armax_siso[0]
    Bo = test_polynomials_armax_siso[1]
    Co = test_polynomials_armax_siso[2]
    N = 1000                 # Number of samples
    u = -1 + 2*randn(N, 1)   # Defines input signal
    e = 0.01*randn(N, 1)    # Emulates gaussian white noise with std = 0.01

    # Calculates the y ARMAX: G(q) = B(q)/A(q) and H(q) = C(q)/A(q)
    y = lfilter(Bo, Ao, u, axis=0) + lfilter(Co, Ao, e, axis=0)

    return [u, y]

def test_armax_siso(test_signals_armax_siso, test_polynomials_armax_siso):
    u = test_signals_armax_siso[0]
    y = test_signals_armax_siso[1]
    A = array(test_polynomials_armax_siso[0])
    B = array(test_polynomials_armax_siso[1])
    C = array(test_polynomials_armax_siso[2])
    t0 = array(A[1:].tolist() + B[1:].tolist() + C[1:].tolist())
    
    nk = 1
    m = armax(len(A), len(B)-(nk+1), len(C)-1, nk , u, y)
    t = m.parameters
    chivalue = get_value_elipse(t, t0, inv(m.P)) #calc a elipse
    
    assert check_inside_elipse(chivalue, len(t)) # verifica se o theta esta dentro da elipse

#SIMO
@pytest.fixture
def test_polynomials_armax_simo(): #isso aqui define A e B para os testes q receberem test_polynomials
    A1o  = [1, -1.2, 0.36]
    A12o = [0, 0.09, -0.1]
    A2o  = [1, -1.6, 0.64]
    A21o = [0, 0.2, -0.01]
    B1o = [0, 0.5, 0.4]
    B2o = [0, 0.2,-0.3]
    C1o  = [1, 0.8,-0.1]
    C2o  = [1, 0.9,-0.2]
    return [A1o,A12o,A2o,A21o,B1o,B2o,C1o,C2o]

@pytest.fixture
def test_signals_armax_simo(test_polynomials_armax_simo):
    A1o  = array(test_polynomials_armax_simo[0])
    A12o = array(test_polynomials_armax_simo[1])
    A2o  = array(test_polynomials_armax_simo[2])
    A21o = array(test_polynomials_armax_simo[3])
    B1o  = array(test_polynomials_armax_simo[4])
    B2o  = array(test_polynomials_armax_simo[5])
    C1o  = array(test_polynomials_armax_simo[6])
    C2o  = array(test_polynomials_armax_simo[7])
    N = 400
    nu = 1
    ny = 2
    #Take u as uniform
    u = -1.5 + 2*1.5*rand(N, nu)
    #Generate gaussian white noise with standat deviation 0.01
    e = 0.01*randn(N, ny)
    #Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
    y1 = zeros((N, 1))
    y2 = zeros((N, 1))
    v1 = lfilter(C1o, [1], e[:,0:1], axis=0)
    v2 = lfilter(C2o, [1], e[:,1:2], axis=0)
    #Simulate the true process 
    for i in range(2, N):
        y1[i] = -dot(A1o[1:3] ,y1[i-2:i][::-1]) - dot(A12o[1:3],y2[i-2:i][::-1]) + dot(B1o[1:3], u[i-2:i, 0][::-1])
        y2[i] = -dot(A21o[1:3], y1[i-2:i][::-1]) - dot(A2o[1:3], y2[i-2:i][::-1]) + dot(B2o[1:3], u[i-2:i, 0][::-1])
    y = concatenate((y1+v1, y2+v2), axis=1)
    return [u,y]

def test_armax_simo(test_signals_armax_simo, test_polynomials_armax_simo):
    A1o  = array(test_polynomials_armax_simo[0])
    A12o = array(test_polynomials_armax_simo[1])
    A2o  = array(test_polynomials_armax_simo[2])
    A21o = array(test_polynomials_armax_simo[3])
    B1o  = array(test_polynomials_armax_simo[4])
    B2o  = array(test_polynomials_armax_simo[5])
    C1o  = array(test_polynomials_armax_simo[6])
    C2o  = array(test_polynomials_armax_simo[7])
    to = array(A1o[1:].tolist(),A12o[1:].tolist(),A21o[1:].tolist(),A2o[1:].tolist(),\
               B1o[1:].tolist(),B2o[1:].tolist(), \
               C1o[1:].tolist(),C2o[1:].tolist())

    u = array(test_signals_armax_simo[0])
    y = array(test_signals_armax_simo[1])
    nk = [[1], [1]]
    na = [[len(A1o)-1,len(A12o)-1],[len(A21o)-1,len(A2o)-1]]
    nb = [[len(B1o)-(nk[0][0]+1)],[len(B2o)-(nk[1][0]+1)]]
    nc = [[len(C1o)-1],[len(C2o)-1]]
    m = armax(na,nb,nc,nk,u,y)
    t = m.parameters
    chiv = get_value_elipse(t,to,inv(m.P))

    assert check_inside_elipse(chiv,len(t))

#MISO
@pytest.fixture
def test_polynomials_armax_miso(): #isso aqui define A e B para os testes q receberem test_polynomials
    Ao  = [1, -1.2, 0.36]
    B0o = [0, 0.5, 0.4]
    B1o = [0, 0.2,-0.3]
    Co  = [1, 0.8,-0.1]
    return [Ao,B0o,B1o,Co]

@pytest.fixture
def test_signals_armax_miso(test_polynomials_armax_miso):
    Ao  = array(test_polynomials_armax_miso[0])
    B0o = array(test_polynomials_armax_miso[1])
    B1o = array(test_polynomials_armax_miso[2])
    Co  = array(test_polynomials_armax_miso[3])
    
    nu = 2
    ny = 1
    N = 200
    u = -1 + 2*rand(N, nu)
    e = 0.01*randn(N, ny)
    y = lfilter(B0o, Ao, u[:,0:1], axis=0) + lfilter(B1o, Ao, u[:,1:2], axis=0) + lfilter(Co, Ao, e[:,0:1], axis=0)
    return[u,y]

def test_armax_miso(test_polynomials_armax_miso,test_signals_armax_miso):
    Ao  = array(test_polynomials_armax_miso[0])
    B0o = array(test_polynomials_armax_miso[1])
    B1o = array(test_polynomials_armax_miso[2])
    Co  = array(test_polynomials_armax_miso[3])
    to = array(Ao[1:].tolist(),\
               B0o[1:].tolist(),B1o[1:].tolist(), \
               Co[1:].tolist())

    u = array(test_signals_armax_miso[0])
    y = array(test_signals_armax_miso[1])
    nk = [[1], [1]]
    na = [[len(Ao)-1]]
    nb = [[len(B0o)-(nk[0][0]+1)],[len(B1o)-(nk[1][0]+1)]]
    nc = [[len(Co)-1]]
    m = armax(na,nb,nc,nk,u,y)
    t = m.parameters
    chiv = get_value_elipse(t,to,inv(m.P))

    assert check_inside_elipse(chiv,len(t))

#MIMO
@pytest.fixture
def test_polynomials_armax_mimo():
    A11o = array(([-1.2, 0.36]))
    A12o = array(([-0.2, 0.1]))
    A21o = array(([-0.05, 0.09]))
    A22o = array(([-1.4, 0.49]))

    B11o = array(([0.5, 0.1]))
    B12o = array(([1.0, 0.66]))
    B21o = array(([0.8, 0.3]))
    B22o = array(([0.65,0.2]))

    C1o  = array([1, 0.8,-0.1])
    C2o  = array([1, 0.9,-0.2])

    return [A11o,A12o,A21o,A22o,B11o,B12o,B21o,B22o,C1o,C2o]

@pytest.fixture
def test_signals_armax_mimo(test_polynomials_armax_mimo):
    # makes mimo signals with S (ARX: G(q) = B(q)/A(q) and H(q) = C/A(q))
    A11o = test_polynomials_armax_mimo[0]
    A12o = test_polynomials_armax_mimo[1]
    A21o = test_polynomials_armax_mimo[2]
    A22o = test_polynomials_armax_mimo[3]
    B11o = test_polynomials_armax_mimo[4]
    B12o = test_polynomials_armax_mimo[5]
    B21o = test_polynomials_armax_mimo[6]
    B22o = test_polynomials_armax_mimo[7]
    C1o = test_polynomials_armax_mimo[8]
    C2o = test_polynomials_armax_mimo[9]
    u = randn(1000, 1)
    e = 0.01*randn(1000, 1)
    
    
    det = convolve(A11o, A22o) - convolve(A12o, A21o) 
    y1 = lfilter(convolve(A22o, B11o), det, u[:, 0:1], axis=0) + \
     lfilter(convolve(-A12o, B21o), det, u[:, 0:1], axis=0) + \
     lfilter(convolve(A22o, B12o), det, u[:, 1:2], axis=0) + \
     lfilter(convolve(-A12o, B22o), det, u[:, 1:2], axis=0) + \
     lfilter(convolve(A22o, C1o), det, e[:, 0:1], axis=0) + \
     lfilter(convolve(-A12o, C2o), det, e[:, 1:2], axis=0)
    y2 = lfilter(convolve(-A21o, B11o), det, u[:, 0:1], axis=0) + \
     lfilter(convolve(A11o, B21o), det, u[:, 0:1], axis=0) + \
     lfilter(convolve(-A21o, B12o), det, u[:, 1:2], axis=0) + \
     lfilter(convolve(A11o, B22o), det, u[:, 1:2], axis=0) + \
     lfilter(convolve(-A21o, C1o), det, e[:, 0:1], axis=0) + \
     lfilter(convolve(A11o, C2o), det, e[:, 1:2], axis=0)

    y = concatenate((y1, y2), axis=1)
    return [u,y]

def test_armax_mimo(test_polynomials_armax_mimo,test_signals_armax_mimo):
    A11o = test_polynomials_armax_mimo[0]
    A12o = test_polynomials_armax_mimo[1]
    A21o = test_polynomials_armax_mimo[2]
    A22o = test_polynomials_armax_mimo[3]
    B11o = test_polynomials_armax_mimo[4]
    B12o = test_polynomials_armax_mimo[5]
    B21o = test_polynomials_armax_mimo[6]
    B22o = test_polynomials_armax_mimo[7]
    C1o = test_polynomials_armax_mimo[8]
    C2o = test_polynomials_armax_mimo[9]
    to = array(A11o[1:].tolist(),A12o[1:].tolist(),A21o[1:].tolist(),A22o[1:].tolist(),\
               B11o[1:].tolist(),B12o[1:].tolist(),B21o[1:].tolist(),B22o[1:].tolist(), \
               C1o[1:].tolist(),C2o[1:].tolist())

    u = array(test_signals_armax_mimo[0])
    y = array(test_signals_armax_mimo[1])
    nk = [[1, 1], [1, 1]]
    na = [[len(A11o)-1,len(A12o)-1],[len(A21o)-1,len(A22o)-1]]
    nb = [[len(B11o)-(nk[0][0]+1),len(B12o)-(nk[0][1]+1)],[len(B21o)-(nk[1][0]+1),len(B22o)-(nk[1][1]+1)]]
    nc = [[len(C1o)-1],[len(C2o)-1]]
    m = armax(na,nb,nc,nk,u,y)
    t = m.parameters

    chiv = get_value_elipse(t,to,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

# -----------------Recursive Module-----------------
def test_rls_siso(test_signals_arx_siso,test_polynomials_arx_siso):
    A = array(test_polynomials_arx_siso[0])
    B = array(test_polynomials_arx_siso[1])
    t0 = array(A[1:].tolist() + B[1:].tolist())

    u = test_signals_arx_siso[0]
    y = test_signals_arx_siso[1]
    nk = 1

    m = rls(len(A), len(B)-(nk+1), nk , u, y)
    t = m.parameters
    chivalue = get_value_elipse(t, t0, inv(m.P)) #calc a elipse

    assert check_inside_elipse(chivalue, len(t)) # verifica se o theta esta dentro da elipse
# ----------------- BJ -----------------
# SISO
@pytest.fixture
def test_polynomials_bj_siso():
    Fo  = [1, -1.2, 0.36]
    Bo = [0, 0.5, 0.1]
    Co = [1, 0.8, 0.2]
    Do = [1, -1.6, 0.64]

    return [Fo, Bo, Co, Do]

@pytest.fixture
def test_signals_bj_siso(test_polynomials_bj_siso):
    Fo = array(test_polynomials_bj_siso[0])
    Bo = array(test_polynomials_bj_siso[1])
    Co = array(test_polynomials_bj_siso[2])
    Do = array(test_polynomials_bj_siso[3])
    N = 400
    u = -1 + 2*randn(N, 1)
    e = 0.1*randn(N, 1)
    y = lfilter(Bo, Fo, u, axis=0) + lfilter(Co, Do, e, axis=0)
    return [u,y]

def test_bj_siso(test_signals_bj_siso,test_polynomials_bj_siso):
    u = test_signals_bj_siso[0]
    y = test_signals_bj_siso[1]
    
    Fo = array(test_polynomials_bj_siso[0])
    Bo = array(test_polynomials_bj_siso[1])
    Co = array(test_polynomials_bj_siso[2])
    Do = array(test_polynomials_bj_siso[3])
    nk = 1
    e = 0.1*randn(len(u), 1)

    m = bj(len(Bo)-(nk+1),len(Co),len(Do),len(Fo),nk,u,y)
    yhat = lfilter(m.B[0,0], m.F[0,0], u, axis=0) + lfilter(m.C[0,0],m.D[0,0], e, axis=0)
    # TODO

#SIMO
@pytest.fixture
def test_polynomials_bj_simo():
    F1o = [1, -1.2, 0.36]
    F2o = [1, -1.4, 0.49]
    B1o = [0, 0.5, 0.1]
    B2o = [0, 0.3,-0.2]
    C1o = [1, 0.8, 0.16]
    C2o = [1, 0.9, 0.22]
    D1o = [1, -1.8, 0.91]
    D2o = [1, -1.6, 0.80]

    return [F1o, F2o, B1o, B2o, C1o, C2o, D1o, D2o]

@pytest.fixture
def test_signals_bj_simo(test_polynomials_bj_simo):
    F1o = array(test_polynomials_bj_simo[0])
    F2o = array(test_polynomials_bj_simo[1])
    B1o = array(test_polynomials_bj_simo[2])
    B2o = array(test_polynomials_bj_simo[3])
    C1o = array(test_polynomials_bj_simo[4])
    C2o = array(test_polynomials_bj_simo[5])
    D1o = array(test_polynomials_bj_simo[6])
    D2o = array(test_polynomials_bj_simo[7])

    N = 400
    u = -1 + 2*randn(N, len(test_polynomials_bj_simo)/4)
    e = 0.01*randn(N, 1)
    y1 = lfilter(B1o, F1o, u[:,0:1], axis=0) + lfilter(C1o, D1o, e[:,0:1], axis=0)
    y2 = lfilter(B2o, F2o, u[:,0:1], axis=0) + lfilter(C2o, D2o, e[:,1:2], axis=0)
    y = concatenate((y1, y2), axis=1)

    return [u,y]

def test_bj_simo(test_polynomials_bj_simo,test_signals_bj_simo):
    y = array(test_signals_bj_simo[1])
    u = array(test_signals_bj_simo[0])

    F1o = array(test_polynomials_bj_simo[0])
    F2o = array(test_polynomials_bj_simo[1])
    B1o = array(test_polynomials_bj_simo[2])
    B2o = array(test_polynomials_bj_simo[3])
    C1o = array(test_polynomials_bj_simo[4])
    C2o = array(test_polynomials_bj_simo[5])
    D1o = array(test_polynomials_bj_simo[6])
    D2o = array(test_polynomials_bj_simo[7])

    nk = [[1],[1]]
    nf = [[len(F1o)-1],[len(F2o)-1]]
    nb = [[len(B1o)-(nk[0,0]+1)],[len(B2o)-(nk[0,1]+1)]]
    nc = [[len(C1o)-1],[len(C2o)-1]]
    nd = [[len(D1o)-1],[len(D2o)-1]]
    e = 0.01*randn(len(u[:,0]), 1)

    m = bj(nb,nc,nd,nf,nk,u,y)
    #Parece que uma saida não interfere na outra, faz sentido?
    y1 = lfilter(m.B[0,0], m.F[0,0], u[:,0:1], axis=0) + lfilter(m.C[0,0], m.D[0,0], e[:,0:1], axis=0)
    y2 = lfilter(m.B[1,0], m.F[1,0], u[:,0:1], axis=0) + lfilter(m.C[1,0], m.B[1,0], e[:,1:2], axis=0)
    yhat = concatenate((y1, y2), axis=1)
    #TODO

# MISO

@pytest.fixture
def test_polynomials_bj_miso():
    F1o = [1, -1.2, 0.36]
    F2o = [1, -1.4, 0.49]
    B1o = [0, 0.5, 0.1]
    B2o = [0, 0.3,-0.2]
    Co = [1, 0.8, 0.16]
    Do = [1, -1.8, 0.91]
    
    return [F1o,F2o,B1o,B2o,Co,Do]

@pytest.fixture
def test_signals_bj_miso(test_polynomials_bj_miso):
    F1o = array(test_polynomials_bj_miso[0])
    F2o = array(test_polynomials_bj_miso[1])
    B1o = array(test_polynomials_bj_miso[2])
    B2o = array(test_polynomials_bj_miso[3])
    Co  = array(test_polynomials_bj_miso[4])
    Do  = array(test_polynomials_bj_miso[5])

    N = 400
    u = -1 + 2*randn(N, (len(test_polynomials_bj_miso) - 2)/2)
    e = 0.01*randn(N, 1)
    y1 = lfilter(B1o, F1o, u[:,0:1], axis=0)
    y2 = lfilter(B2o, F2o, u[:,1:2], axis=0)
    ye = lfilter(Co, Do, e[:,0:1], axis=0)
    y = y1 + y2 + ye

    return [u,y]

def test_bj_miso(test_polynomials_bj_miso,test_signals_bj_miso):
    F1o = array(test_polynomials_bj_miso[0])
    F2o = array(test_polynomials_bj_miso[1])
    B1o = array(test_polynomials_bj_miso[2])
    B2o = array(test_polynomials_bj_miso[3])
    # Co  = array(test_polynomials_bj_miso[4])
    # Do  = array(test_polynomials_bj_miso[5])
    y = test_signals_bj_miso[1]
    u = test_signals_bj_miso[0]

    nk = [[1,1]]
    nf = [[len(F1o)-1,len(F2o)-1]]
    nb = [[len(B1o)-(nk[0,0]+1),len(B2o)-(nk[0,1]+1)]]
    nc = [[2]]
    nd = [[2]]
    e = 0.01*randn(len(u[:,0]), 1)

    m = bj(nb,nc,nd,nf,nk,u,y)
    yhat = lfilter(m.B[0,0], m.F[0,0], u[:,0:1], axis=0) + lfilter(m.B[1,0], m.F[1,0], u[:,1:2], axis=0) + lfilter(m.C[0,0], m.D[0,0], e[:,0:1], axis=0)
    #TODO

#MIMO
@pytest.fixture
def test_polynomials_bj_mimo():
    F11o = [1, -1.2, 0.36]
    F12o = [1, -1.8, 0.91]
    F21o = [1, -1.4, 0.49]
    F22o = [1, -1.0, 0.25]
    B11o = [0, 0.5, 0.1]
    B12o = [0, 0.9,-0.1]
    B21o = [0, 0.4, 0.8]
    B22o = [0, 0.3,-0.2]
    C1o = [1, 0.8, 0.16]
    C2o = [1, 0.9, 0.22]
    D1o = [1, -1.8, 0.91]
    D2o = [1, -1.6, 0.80]
    return [F11o,F12o,F21o,F22o,B11o,B12o,B21o,B22o,C1o,C2o,D1o,D2o]

@pytest.fixture
def test_signals_bj_mimo(test_polynomials_bj_mimo):
    F11o = array(test_polynomials_bj_mimo[0])
    F12o = array(test_polynomials_bj_mimo[1])
    F21o = array(test_polynomials_bj_mimo[2])
    F22o = array(test_polynomials_bj_mimo[3])
    B11o = array(test_polynomials_bj_mimo[4])
    B12o = array(test_polynomials_bj_mimo[5])
    B21o = array(test_polynomials_bj_mimo[6])
    B22o = array(test_polynomials_bj_mimo[7])
    C1o  = array(test_polynomials_bj_mimo[8])
    C2o  = array(test_polynomials_bj_mimo[9])
    D1o  = array(test_polynomials_bj_mimo[10])
    D2o  = array(test_polynomials_bj_mimo[11])
    N = 400
    nu = 2
    ny = 2
    u = -1 + 2*randn(N, nu)
    e = 0.01*randn(N, ny)
    y1 = lfilter(B11o, F11o, u[:,0:1], axis=0) + lfilter(B12o, F12o, u[:,1:2], axis=0) + lfilter(C1o, D1o, e[:,0:1], axis=0)
    y2 = lfilter(B21o, F21o, u[:,0:1], axis=0) + lfilter(B22o, F22o, u[:,1:2], axis=0) + lfilter(C2o, D2o, e[:,1:2], axis=0)
    y = concatenate((y1, y2), axis=1)

    return [u,y]

def test_bj_mimo(test_signals_bj_mimo,test_polynomials_bj_mimo):
    F11o = array(test_polynomials_bj_mimo[0])
    F12o = array(test_polynomials_bj_mimo[1])
    F21o = array(test_polynomials_bj_mimo[2])
    F22o = array(test_polynomials_bj_mimo[3])
    B11o = array(test_polynomials_bj_mimo[4])
    B12o = array(test_polynomials_bj_mimo[5])
    B21o = array(test_polynomials_bj_mimo[6])
    B22o = array(test_polynomials_bj_mimo[7])
    C1o  = array(test_polynomials_bj_mimo[8])
    C2o  = array(test_polynomials_bj_mimo[9])
    D1o  = array(test_polynomials_bj_mimo[10])
    D2o  = array(test_polynomials_bj_mimo[11])
    
    u = test_signals_bj_mimo[0]
    y = test_signals_bj_mimo[1]
    
    nk = [[1, 1], [1, 1]]
    nf = [[len(F11o)-1,len(F12o)-1],[len(F21o)-1,len(F22o)-1]]
    nb = [[len(B11o)-(nk[0,0]+1),len(B12o)-(nk[0,1]+1)],[len(B21o)-(nk[1,0]+1),len(B22o)-(nk[1,1]+1)]]
    nc = [[len(C1o)-1],[len(C2o)-1]]
    nd = [[len(D1o)-1],[len(D2o)-1]]
    e = 0.01*randn(len(y[:,0]), len(y[0,:]))
    
    m = bj(nb, nc, nd, nf, nk, u, y)
    y1 = lfilter(m.B[0,0], m.F[0,0], u[:,0:1], axis=0) + lfilter(m.B[0,1], m.F[0,1], u[:,1:2], axis=0) + lfilter(m.C[0], m.D[0], e[:,0:1], axis=0)
    y2 = lfilter(m.B[1,0] ,m.F[1,0], u[:,0:1], axis=0) + lfilter(m.B[1,1], m.F[1,1], u[:,1:2], axis=0) + lfilter(m.C[1], m.D[1], e[:,1:2], axis=0)
    yhat = concatenate((y1, y2), axis=1)
    # TODO
# -----------------  -----------------
# Defines sets of arguments for tests that request the respective fixtures
# Here, the values of na to nk are varied
@pytest.mark.parametrize("na, nb, nc, nd, nf, nk", [(1, 0 , 1, 1, 0, 1), (1, 2, 3, 1, 0, 1)])

# ------------------- TEST FUNCTIONS -------------------

# Validates polynomial model orders
# Every passed argument is a requested @pytest.fixture
def test_polynomial_orders(na, nb, nc, nd, nf, nk):
    # Checks the consistency of na 
    assert isinstance(na, int)
    
    # Checks the consistency of nb
    assert isinstance(nb, int)
    
    # Checks the consistency of nc
    assert isinstance(nc, int)

    # Checks the consistency of nk
    assert isinstance(nd, int)

    # Checks the consistency of nk
    assert isinstance(nf, int)

    # Checks the consistency of nk
    assert isinstance(nk, int)

def test_data_type(test_signals_arx_siso):
    u = test_signals_arx_siso[0]
    y = test_signals_arx_siso[1]

    # Checks the consistency of u(t)
    assert isinstance(u, ndarray) or isinstance(u, list)

    # Checks the consistency of y(t)
    assert isinstance(y, ndarray) or isinstance(y, list)

# #@pytest.mark.xfail
# def test_arx(test_signals_arx_siso):
#     # Signals
#     u = test_signals_arx_siso[0]
#     y = test_signals_arx_siso[1]
#     Ao = array(test_signals_arx_siso[2])
#     Bo = array(test_signals_arx_siso[3])

#     A, B = arx(2,1,1,u,y)
    
#     print(A.tolist() == Ao)
#     assert isinstance(A,ndarray)
#     assert isinstance(A.tolist(),list)
#     assert not isinstance(A,list)