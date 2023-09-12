# @author: lima84
"""
    Testing modules for pemethod.py using pytest
"""
import pytest
from numpy import array, ndarray, convolve, cos, sin, concatenate, zeros, dot, \
    sqrt, pi, roots, abs, ones, amax, append, reshape, isnan, isfinite, trim_zeros
from numpy.random import rand, randn, randint
from numpy.linalg import inv
from scipy.signal import lfilter
from pysid.identification.pemethod import arx, armax, bj, oe
from pysid.identification.recursive import rls, els
from pysid.io.print import print_model
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
    return (t - t0) @ P @ (t - t0).T

def check_inside_elipse(chivalue, df, alfa=0.995):
    return chivalue < chi2.ppf(alfa, df=df)

def gen_stable_poly(order):
    istable = False
    A = [1]
    k = 1
    while not istable:
        # Decide whether will be complex or not
        if order - k == 1:
            k += 1
            A = convolve(A, [1,  1-2*rand()])
        else:
            if rand() > 0.6:
                t = rand()
                u = rand()

                x = sqrt(t) * cos(2*pi*u)
                y = sqrt(t) * sin(2*pi*u)
                A = convolve(A, [1, -2*x, x**2 + y**2])
                k += 2
            else:
                A = convolve(A, [1,  1-2*rand()])
                k += 1

        if k >= order:
            # Check if A is stable
            r = roots(A)
            if any(abs(r) > 1):
                A = [1]
                k = 1
            else:
                istable = True

    return A

def gen_stable_model2x2(na):
    stable = False
    i = 1
    A = zeros((2,2),dtype=object)
    na = na + ones(na.shape,dtype=int)
    while not stable:
        A = zeros((2,2),dtype=object)
        for i in range(2):
            for j  in range(2):
                if i == j:
                    A[i,j] = gen_stable_poly(na[i,j])
                else:
                    A[i,j] = gen_stable_poly(na[i,j])
                    A[i,j][0] = 0
                    while not all(r < 1.0 for r in abs(roots(A[i,j]))):
                        A[i,j] = gen_stable_poly(na[i,j])
                        A[i,j][0] = 0

        if(len(convolve(A[0,0], A[1,1]))>len(convolve(A[0,1], A[1,0]))):
            det = convolve(A[0,0], A[1,1]) - \
                concatenate(([0]*abs(len(convolve(A[0,1], A[1,0]))-len(convolve(A[0,0], A[1,1]))),convolve(A[0,1], A[1,0])))
        else:
            det = concatenate(([0]*abs(len(convolve(A[0,1], A[1,0]))-len(convolve(A[0,0], A[1,1]))),convolve(A[0,0], A[1,1]))) - \
                convolve(A[0,1], A[1,0])

        if all(r < 1.0 for r in abs(roots(det))):
            stable = True
    print(stable)
    return A#,trim_zeros(det,'f')

# -----------------Arx-----------------

@pytest.fixture
def test_polynomials_arx_siso():
    Ao = [1, -1.2, 0.36]
    Bo = [0, 0.5, 0.1]

    return [Ao, Bo]

@pytest.fixture
def test_signals_arx_siso(test_polynomials_arx_siso):
    # True test system parameters
    Ao = test_polynomials_arx_siso[0]
    Bo = test_polynomials_arx_siso[1]

    # Replicates the following experiment:
    # y(t) = Go(q)*u(t) + Ho(q)*e(t),
    # where u(t) is the system input and e(t) white noise
    N = 1000                              # Number of samples
    u = -sqrt(3) + 2*sqrt(3)*rand(N, 1)   # Defines input signal
    e = 0.01*randn(N, 1)                  # Emulates Gaussian white noise with std = 0.01

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = lfilter(Bo, Ao, u, axis=0) + lfilter([1], Ao, e, axis=0)

    return [u, y]

# Test with known parameter
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

def test_arx_miso():

    ny = 1        #number of outputs (single)
    nu = 2        #number of inputs

    na = 2
    nb = array([[1, 1]])
    nk = array([[1, 1]])

    Ao = zeros((ny,ny),dtype=object)
    Bo = zeros((ny,nu),dtype=object)
    Ao[0,0] = [1, -1.2, 0.36]
    Bo[0,0] = [0, 1.0, 0.5]
    Bo[0,1] = [0, 0.8, 0.3]

    N = 1000                # Number of samples
    e = 0.1*randn(N, 1)    # Emulates Gaussian white noise with std = 0.01

    u = zeros((N,nu), dtype=float)
    for i in range(nu):
        u[:,i] = -sqrt(3) + 2*sqrt(3)*rand(N,)

    t0 = Ao[0,0][1:]
    y = lfilter([1], Ao[0,0], e, axis=0)
    #simulate the model and makes t0
    for i in range(nu):
        y = y + lfilter(Bo[0,i], Ao[0,0], u[:,i]).reshape(N,1)
        t0 = concatenate((t0, Bo[0,i][nk[0,i]:]),axis=0)

    m = arx(na,nb,nk,u,y)
    t = m.parameters
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

def test_arx_simo():
    nu = 1
    ny = 2

    na = array([[2, 2], [2, 2]])
    nb = array([[1], [1]])
    nk = array([[1], [1]])

    Ao = zeros((ny,ny),dtype=object)
    Bo = zeros((ny,nu),dtype=object)
    Ao[0,0] = [1, -1.2, 0.36]
    Ao[0,1] = [0, 0.04,-0.05]
    Ao[1,0] = [0, 0.09, 0.03]
    Ao[1,1] = [1, -1.6, 0.64]
    Bo[0,0] = [0, 1.0, 0.5]
    Bo[1,0] = [0, 0.8, 0.3]

    N = 1000                              # Number of samples
    u = -sqrt(3) + 2*sqrt(3)*rand(N, 1)   # Defines input signal
    e = 0.01*randn(N, ny)                  # Emulates Gaussian white noise with std = 0.01

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = zeros((N,ny),dtype=float) #y's, com ny linhas e N colunas, cada linha é uma saida
    L = max(amax(na),amax(nb+nk)) #to know were to start
    for i in range(L,N):
        for j in range(ny): # for each output
            for k in range(ny): # to travel in cols of the Ao matrix
                # [::-1] makes the array backwards
                y[i,j] += dot(Ao[j,k][1:],-y[i-(len(Ao[j,k])-1):i,k][::-1])
            y[i,j] += dot(Bo[j,0][nk[j,0]:],u[i-len(Bo[j,0][nk[j,0]:]):i,0][::-1])
        y[i,j] += e[i,j]

    t0 = array([])
    for i in range(ny):
        for j in range(ny):
            t0 = concatenate((t0, Ao[i,j][1:]))
    print("last 'A' param -->",t0[-1:],"\n")
    for i in range(ny):
        t0 = concatenate((t0, Bo[i,0][nk[i,0]:]))
    m = arx(na,nb,nk,u,y)
    t = m.parameters
    print(t)
    print(t0)
    for i in range(len(t0)):
        print(t0[i],"\t##\t",t[i])
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

def test_arx_mimo():
    nu = 2
    ny = 2

    na = array([[2, 2], [2, 2]])
    nb = array([[1, 1], [1, 1]])
    nk = array([[1, 1], [1, 1]])

    Ao = zeros((ny,ny),dtype=object)
    Bo = zeros((ny,nu),dtype=object)
    Ao[0,0] = [1, -1.2, 0.36]
    Ao[0,1] = [0, 0.2, 0.1]
    Ao[1,0] = [0, -0.05, 0.09]
    Ao[1,1] = [1, -1.4, 0.49]

    Bo[0,0] = [0, 0.5, 0.1]
    Bo[0,1] = [0, 1, 0.66]
    Bo[1,0] = [0, 0.8, 0.3]
    Bo[1,1] = [0, 0.65, 0.2]

    N = 1000                # Number of samples
    e = 0.01*randn(N, ny)    # Emulates Gaussian white noise with std = 0.01
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(nu):
        for j in range(ny):
            u[:,i] = -sqrt(3) + 2*sqrt(3)*rand(N,)

    y = zeros((N,ny),dtype=float) #y's, com ny linhas e N colunas, cada linha é uma saida
    L = max(amax(na),amax(nb+nk)) #to know were to start
    for i in range(L,N):
        for j in range(ny): # for each output
            for k in range(ny): # to travel in cols of the Ao matrix
                # [::-1] makes the array backwards
                y[i,j] += dot(Ao[j,k][1:],-y[i-(len(Ao[j,k])-1):i,k][::-1])
            for k in range(nu):# for each input
                y[i,j] += dot(Bo[j,k][nk[j,k]:],u[i-len(Bo[j,k][nk[j,k]:]):i,k][::-1])
        y[i,j] += e[i,j]

    t0 = array([])
    for i in range(ny):
        for j in range(ny):
            t0 = concatenate((t0, Ao[i,j][1:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Bo[i,j][nk[i,j]:]))
    m = arx(na,nb,nk,u,y)
    t = m.parameters
    chivalue = get_value_elipse(t, t0, inv(m.P))

    assert check_inside_elipse(chivalue, len(t))

# Random test
def test_arx_random_siso():
    # Test signals
    na = 1 + randint(0, 50)
    nb = min(1 + randint(0, 50), na)
    nk = 1 + randint(0, 50)

    # Generate polynomials
    Ao = gen_stable_poly(na)
    Boo = -1 + 2*randn(nb)
    Bo = array([0,]*nk + Boo.tolist())
    
    N = 1000                              # Number of samples
    u = -sqrt(3) + 2*sqrt(3)*rand(N, 1)   # Defines input signal
    e = 0.01*randn(N, 1)                  # Emulates Gaussian white noise with std = 0.01

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = lfilter(Bo, Ao, u, axis=0) + lfilter([1], Ao, e, axis=0)

    t0 = array(Ao[1:].tolist() + Bo[nk:].tolist())

    m = arx(na-1, nb-1, nk, u, y)
    t = m.parameters
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

def test_arx_random_simo():
    n = 10
    ny = 2 #randint(2,n)                  #number of outputs
    nu = 1                             #number of inputs (single)

    na = randint(2,n,(ny,ny)) #ny x ny
    # na = array([[10, 10], [10, 10]]) #ok, não é um problema

    #Generate Polynomials
    # Ao = zeros((ny,ny),dtype=object)
    # for i in  range(ny):
    #     for j in range(ny):
    #         Ao[i,j] = gen_stable_poly(na[i,j])
    #         if i != j:
    #             Ao[i,j][0] = 0
    #             while not all(r < 1.0 for r in abs(roots(Ao[i,j]))):
    #                 Ao[i,j] = gen_stable_poly(na[i,j])
    #                 Ao[i,j][0] = 0
    Ao = gen_stable_model2x2(na)

    Bo = zeros((ny,1), dtype=object)
    nk = array([[randint(1, n)], [randint(1, n)]]) #1 + randint(0, n,(ny,nu))
    nb = array([],dtype=int)
    for i in range(ny):
        nb = append(nb,1+randint(5,n))
        Boo = -1 + 2*randn(nb[i])
        Bo[i][0] = concatenate(([0,]*nk[i,0],Boo))

    nb = reshape(nb, (ny,nu)) #from (ny,) to (ny,1) 
    N = 1000                              # Number of samples
    u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)   # Defines input signal
    e = 0.025*randn(N, ny)                  # Emulates Gaussian white noise with std = 0.01

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = zeros((N,ny),dtype=float) #y's, com N rows and ny columns
    L = max(amax(na),amax(nb+nk)) #to know were to start
    for i in range(L,N):
        for j in range(ny): # for each output
            for k in range(ny): # to travel in cols of the Ao matrix
                # [::-1] makes the array backwards
                y[i,j] += dot(Ao[j,k][1:],-y[i-(len(Ao[j,k])-1):i,k][::-1])
            for k in range(nu):# for each input
                y[i,j] += dot(Bo[j,k][nk[j,k]:],u[i-len(Bo[j,k][nk[j,k]:]):i,k][::-1])
            y[i,j] += e[i,j]

    t0 = array([])
    for i in range(ny):
        for j in range(ny):
            t0 = concatenate((t0, Ao[i,j][1:]))
    for i in range(ny):
        t0 = concatenate((t0, Bo[i,0][nk[i,0]:]))
    m = arx(na,nb-ones(nb.shape,dtype=int),nk,u,y)
    t = m.parameters
    # for i in range(ny):
    #     for j in range(ny):
    #         print(abs(roots(Ao[i,j])))
    print(na,"\n",nk)
    print(t0.shape,t.shape)
    for i in range(len(t)):
        print(t[i],"\t ## \t", t0[i])
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))


def test_arx_random_miso():
    n = 10
    ny = 1                   #number of outputs (single)
    nu = randint(2,n)        #number of inputs

    nb = randint(2,n,(ny,nu)) #1 x nu
    na = max(nb.max(), 1+randint(0,10))
    nk = 1 + randint(0, n,(ny,nu))

    #Generate Polynomials
    Ao = gen_stable_poly(na)

    N = 1000                # Number of samples
    e = 0.01*randn(N, 1)    # Emulates Gaussian white noise with std = 0.01

    Bo = zeros((nu,1), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(nu):
        Boo = -1 + 2*randn(nb[0,i]) #generate B's
        Bo[i][0] = concatenate(([0,]*nk[0,i],Boo))
        u[:,i] = -sqrt(3) + 2*sqrt(3)*rand(N,)

    t0 = Ao[1:].tolist()

    y = lfilter([1], Ao, e, axis=0)
    #simulate the model and makes t0
    for i in range(nu):
        y = y + lfilter(Bo[i,0], Ao, u[:,i]).reshape(N,1)
        t0 = concatenate((t0, Bo[i,0][nk[0,i]:].tolist()),axis=0)

    m = arx(int(na-1),nb-ones(nb.shape,dtype=int),nk,u,y)
    t = m.parameters
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

def test_arx_random_mimo():
    n = 10
    ny = 2 #randint(2,n)        #number of outputs (single)
    nu = 2 #randint(2,n)        #number of inputs

    nb = 1 + randint(1,n,(ny,nu)) #ny x nu
    na = 1 + randint(1,n,(ny,ny)) #ny x ny
    nk = array([[1, 1], [1, 1]]) #1 + randint(0, n,(ny,nu)) 
    #nk[i,j] delay of uj input for yi output
    #nb[i,j] order of uj input for yi output

    N = 1000                # Number of samples
    e = 0.01*randn(N, ny)    # Emulates Gaussian white noise with std = 0.01


    #Generate Polynomials
    # Ao = zeros((ny,ny),dtype=object)
    # for i in  range(ny):
    #     for j in range(ny):
    #         Ao[i,j] = gen_stable_poly(na[i,j])
    #         if i != j:
    #             Ao[i,j][0] = 0
    Ao = gen_stable_model2x2(na)

    Bo = zeros((ny,nu), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(ny):
        for j in range(nu):
            Boo = -1 + 2*randn(nb[i,j]) #generate B's
            Bo[i,j] = concatenate(([0,]*nk[i,j],Boo))

    u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)   # Defines input signal

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = zeros((N,ny),dtype=float) #y's, com ny linhas e N colunas, cada linha é uma saida
    L = max(amax(na),amax(nb+nk)) #to know were to start
    print(Ao)
    for i in range(L,N):
        for j in range(ny): # for each output
            for k in range(ny): # to travel in cols of the Ao matrix
                # [::-1] makes the array backwards
                y[i,j] += dot(Ao[j,k][1:],-y[i-(len(Ao[j,k])-1):i,k][::-1])
            for k in range(nu):# for each input
                y[i,j] += dot(Bo[j,k][nk[j,k]:],u[i-len(Bo[j,k][nk[j,k]:]):i,k][::-1])
            y[i,j] += e[i,j]

    for i in range(ny):
        print(i,": Has NaN: ",isnan(y[:,i]).any()," Has Inf: ",isfinite(y[:,i]).any())
    t0 = array([])
    for i in range(ny):
        for j in range(ny):
            t0 = concatenate((t0, Ao[i,j][1:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Bo[i,j][nk[i,j]:]))
    m = arx(na,nb-ones(nb.shape,dtype=int),nk,u,y)
    t = m.parameters
    # print(Ao)
    # print(na)
    # print(Bo)
    # print(nb)
    # print(m.delay)
    # print(t0.shape,t.shape)
    for i in range(len(t)):
        if i == sum(sum(na)):
            print("--")
        print(t[i],"\t ## \t", t0[i],end="")
        print(f'\t{100*abs(t0[i]-t[i])/t0[i]:.2f}')
        # if abs(t[i]-t0[i])/abs(t0[i]) < 0.25:
        #     print("\tv")
        # else:
        #     print("\tx")
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

# test_arx_random_mimo()
# -----------------Armax-----------------
#SISO
@pytest.fixture
def test_polynomials_armax_siso(): #isso aqui define A e B para os testes q receberem test_polynomials
    Ao = [1, -1.2, 0.36]
    Bo = [0, 0.5, 0.1]
    Co = [1, 0.8, -0.1]
    return [Ao,Bo,Co]

@pytest.fixture
def test_signals_armax_siso(test_polynomials_armax_siso): # isso aqui define u,y, Ao e Bo para os testes que receberem test_signals
    # makes siso signals with S (ARX: G(q) = B(q)/A(q) and H(q) = C/A(q))
    Ao = test_polynomials_armax_siso[0]
    Bo = test_polynomials_armax_siso[1]
    Co = test_polynomials_armax_siso[2]
    N = 1000                              # Number of samples
    u = -sqrt(3) + 2*sqrt(3)*rand(N, 1)   # Defines input signal
    e = 0.01*randn(N, 1)                  # Emulates gaussian white noise with std = 0.01

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
    m = armax(len(A)-1, len(B)-(nk+1), len(C)-1, nk , u, y)

    t = m.parameters
    chivalue = get_value_elipse(t, t0, m.M) #calc a elipse

    
    assert check_inside_elipse(chivalue, len(t)) # verifica se o theta esta dentro da elipse

def test_armax_random_siso():
    n = 10
    ny = 1
    nu = 1
    na = 1 + randint(1, n)
    nb = min(1 + randint(0, n), na)
    nc = 1 + randint(1, n)
    nk = 1 + randint(0, n)

    # Generate polynomials
    Ao = gen_stable_poly(na+1)
    Boo = -1 + 2*randn(nb)
    Bo = array([0,]*nk + Boo.tolist())
    Co = gen_stable_poly(nc+1)

    N = 1000                              # Number of samples
    u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)   # Defines input signal
    e = 0.05*randn(N, ny)                  # Emulates Gaussian white noise with std = 0.01

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = lfilter(Bo, Ao, u, axis=0) + lfilter(Co, Ao, e, axis=0)
    t0 = array(Ao[1:].tolist() + Bo[nk:].tolist() + Co[1:].tolist())

    m = armax(na, nb-1, nc, nk, u, y)
    t = m.parameters
    print("t",len(t),"t0",len(t0))
    for i in range(len(t0)):
        print(t[i],"\t ## \t", t0[i])
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

#SIMO

def test_armax_random_simo():
    n = 5
    ny = 2 #randint(2,n)        #number of outputs (single)
    nu = 1 #randint(2,n)        #number of inputs

    nb =  randint(1,n,(ny,nu)) #ny x nu
    na = randint(1,n,(ny,ny)) #ny x ny
    nc = randint(1,n,(ny,1)) #1 x nu
    nk = array([[1],[1]]) #1 + randint(0, n,(ny,nu))
    #nk[i,j] delay of uj input for yi output
    #nb[i,j] order of uj input for yi output

    N = 1000                # Number of samples
    e = 0.01*randn(N, ny)    # Emulates Gaussian white noise with std = 0.01


    #Generate Polynomials
    # Ao = zeros((ny,ny),dtype=object)
    # for i in  range(ny):
    #     for j in range(ny):
    #         Ao[i,j] = gen_stable_poly(na[i,j])
    #         if i != j:
    #             Ao[i,j][0] = 0
    Ao, det = gen_stable_model2x2(na)

    Bo = zeros((ny,nu), dtype=object)
    Co = zeros((ny,1), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(ny):
        for j in range(nu):
            Boo = -1 + 2*randn(nb[i,j]) #generate B's
            Bo[i,j] = concatenate(([0,]*nk[i,j],Boo))
    for i in range(ny):
        Co[i,0] = gen_stable_poly(nc[i,0]+1)

    u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)   # Defines input signal

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = zeros((N,ny),dtype=float) #y's, com ny linhas e N colunas, cada linha é uma saida
    L = max(amax(na),amax(nb+nk),amax(nc)) #to know were to start
    for i in range(L,N):
        for j in range(ny): # for each output
            for k in range(ny): # to travel in cols of the Ao matrix
                # [::-1] makes the array backwards
                y[i,j] += dot(Ao[j,k][1:],-y[i-(len(Ao[j,k])-1):i,k][::-1])
            for k in range(nu):# for each input
                y[i,j] += dot(Bo[j,k][nk[j,k]:],u[i-len(Bo[j,k][nk[j,k]:]):i,k][::-1])
            y[i,j] += dot(Co[j,0][1:],e[i-len(Co[j,0][1:]):i,j][::-1])
    # print(det)
    # y1 = lfilter(convolve(Ao[1,1], Bo[0,0]), det, u[:, 0:1], axis=0) + \
    #      lfilter(convolve(-Ao[0,1], Bo[1,0]), det, u[:, 0:1], axis=0) + \
    #      lfilter(convolve(Ao[1,1], Co[0,0]), det, e[:, 0:1], axis=0) + \
    #      lfilter(convolve(-Ao[0,1], Co[1,0]), det, e[:, 1:2], axis=0)
    # y2 = lfilter(convolve(-Ao[1,0], Bo[0,0]), det, u[:, 0:1], axis=0) + \
    #      lfilter(convolve(Ao[0,0], Bo[1,0]), det, u[:, 0:1], axis=0) + \
    #      lfilter(convolve(-Ao[1,0], Co[0,0]), det, e[:, 0:1], axis=0) + \
    #      lfilter(convolve(Ao[0,0], Co[1,0]), det, e[:, 1:2], axis=0)

    # y = concatenate((y1, y2), axis=1)

    t0 = array([])
    for i in range(ny):
        for j in range(ny):
            t0 = concatenate((t0, Ao[i,j][1:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Bo[i,j][nk[i,j]:]))
    for i in range(ny):
        t0 = concatenate((t0, Co[i,0][1:]))

    m = armax(na,nb-ones(nb.shape,dtype=int),nc,nk,u,y)
    t = m.parameters
    # print("Ao: ",Ao)
    # print("A: ",m.A,"\n\n")
    # print(Bo)
    # print(Co)
    # print(na,nb,nc)
    # print(t0.shape,t.shape)
    for i in range(len(t)):
        print(t[i],"\t ## \t", t0[i])
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

@pytest.fixture
def test_polynomials_armax_simo():
    A1o  = [1, -1.2, 0.36]
    A12o = [0, 0.09, -0.1]
    A2o  = [1, -1.6, 0.64]
    A21o = [0, 0.2, -0.01]
    B1o = [0, 0.6, 0.4]
    B2o = [0, 0.2,-0.3]
    C1o  = [1, 0.8,-0.1]
    C2o  = [1, 0.7,-0.2]
    return [A1o, A12o, A2o, A21o, B1o, B2o, C1o, C2o]

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
    N = 1000
    nu = 1
    ny = 2
    # Take u as uniform
    u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)
    # Generate gaussian white noise with standat deviation 0.01
    e = 0.01*randn(N, ny)
    # Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
    y1 = zeros((N, 1))
    y2 = zeros((N, 1))
    v1 = lfilter(C1o, [1], e[:,0:1], axis=0)
    v2 = lfilter(C2o, [1], e[:,1:2], axis=0)
    # Simulate the true process
    for i in range(2, N):
        y1[i] = -dot(A1o[1:3] ,y1[i-2:i][::-1]) - dot(A12o[1:3], y2[i-2:i][::-1]) + dot(B1o[1:3], u[i-2:i, 0][::-1])
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
    to = array(A1o[1:].tolist() + A12o[1:].tolist() + A21o[1:].tolist() + A2o[1:].tolist() + \
               B1o[1:].tolist() + B2o[1:].tolist() + \
               C1o[1:].tolist() + C2o[1:].tolist())

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
def test_armax_random_miso():
    n = 15
    nu = 2
    ny = 1

    na = 1 + randint(1, n)
    nb = randint(2,n,(ny,nu)) # 1 x nu
    nc = 1 + randint(1, n)
    nk = array([[randint(1, n),randint(1, n)]]) #+ randint(1, n)
    # Generate polynomials
    Ao = gen_stable_poly(na)
    Co = gen_stable_poly(nc+1)

    N = 1000                # Number of samples
    e = 0.01*randn(N, ny)    # Emulates Gaussian white noise with std = 0.01

    Bo = zeros((nu,1), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(nu):
        Boo = -1 + 2*randn(nb[0,i]) #generate B's
        Bo[i][0] = concatenate(([0,]*nk[0,i],Boo))
        u[:,i] = -sqrt(3) + 2*sqrt(3)*rand(N,)


    y = lfilter(Bo[0,0], Ao, u[:,0:1], axis=0) + lfilter(Bo[1,0], Ao, u[:,1:2], axis=0) + lfilter(Co, Ao, e[:,0:1], axis=0)
    m = armax(na-1,nb-ones(nb.shape,dtype=int),nc,nk,u,y)
    t = m.parameters

    t0 = array([])
    t0 = concatenate((t0, Ao[1:]))
    for i in range(nu):
        t0 = concatenate((t0, Bo[i,0][nk[0,i]:]))
    t0 = concatenate((t0, Co[1:]))
    for i in range(len(t0)):
        print(t[i],"\t ## \t", t0[i])
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

@pytest.fixture
def test_polynomials_armax_miso():
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
    N = 1000
    u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)
    e = 0.01*randn(N, ny)
    y = lfilter(B0o, Ao, u[:,0:1], axis=0) + lfilter(B1o, Ao, u[:,1:2], axis=0) + lfilter(Co, Ao, e[:,0:1], axis=0)
    return[u, y]

def test_armax_miso():
    ny = 1
    nu = 2

    na = 2
    nb = array([[1, 1]])
    nk = array([[1, 1]])
    nc = 2

    Ao = zeros((ny,ny),dtype=object)
    Bo = zeros((ny,nu),dtype=object)
    Co = zeros((ny,1), dtype=object)
    Ao[0,0] = [1, -1.2, 0.36]
    Bo[0,0] = [0, 0.5, 0.4]
    Bo[0,1] = [0, 0.2,-0.3]
    Co[0,0] = [1, 0.8,-0.1]

    N = 1000
    u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)
    e = 0.00001*randn(N, ny)
    t0 = concatenate((Ao[0,0][1:],
               Bo[0,0][1:],Bo[0,1][1:], \
               Co[0,0][1:]))

    u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)
    e = 0.01*randn(N, ny)
    y = lfilter(Bo[0,0], Ao[0,0], u[:,0:1], axis=0) + \
        lfilter(Bo[0,1], Ao[0,0], u[:,1:2], axis=0) + \
        lfilter(Co[0,0], Ao[0,0], e[:,0:1], axis=0)

    m = armax(na,nb,nc,nk,u,y)
    t = m.parameters
    for i in range(len(t0)):
       print(t0[i],"\t##\t",t[i])
    chiv = get_value_elipse(t,t0,inv(m.P))

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

    N = 1000
    u = -sqrt(3) * 2*sqrt(3)*randn(N, 2)
    e = 0.01*randn(N, 2)
    
    
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

def test_armax_mimo():
    ny = 2
    nu = 2

    na = [[2, 2], [2, 2]]
    nb = [[1, 1], [1, 1]]
    nk = [[1, 1], [1, 1]]
    nc = [[2], [2]]

    Ao = array([[[1, -1.2, 0.36], [0, 0.09, -0.1]],
                [[0, 0.2, -0.01],[1, -1.6, 0.64]]])

    Bo = array([[[0, 0.5, 0.4], [0, 0.9, 0.8]],
                [[0, 0.2,-0.3], [0, 0.1,-0.8]]])

    Co = array([[[1, 0.8,-0.1]],
                [[1, 0.9,-0.2]]])

    N = 1000
    # Take u as uniform
    u = -1 + 2*rand(N, nu)
    # Generate gaussian white noise with standat deviation 0.01
    e = 0.01*randn(N, ny)

    det = convolve(Ao[0,0], Ao[1,1]) - convolve(Ao[0,1], Ao[1,0]) 
    y1 = lfilter(convolve(Ao[1,1], Bo[0,0]), det, u[:, 0:1], axis=0) + \
         lfilter(convolve(-Ao[0,1], Bo[1,0]), det, u[:, 0:1], axis=0) + \
         lfilter(convolve(Ao[1,1], Bo[0,1]), det, u[:, 1:2], axis=0) + \
         lfilter(convolve(-Ao[0,1], Bo[1,1]), det, u[:, 1:2], axis=0) + \
         lfilter(convolve(Ao[1,1], Co[0,0]), det, e[:, 0:1], axis=0) + \
         lfilter(convolve(-Ao[0,1], Co[1,0]), det, e[:, 1:2], axis=0)
    y2 = lfilter(convolve(-Ao[1,0], Bo[0,0]), det, u[:, 0:1], axis=0) + \
         lfilter(convolve(Ao[0,0], Bo[1,0]), det, u[:, 0:1], axis=0) + \
         lfilter(convolve(-Ao[1,0], Bo[0,1]), det, u[:, 1:2], axis=0) + \
         lfilter(convolve(Ao[0,0], Bo[1,1]), det, u[:, 1:2], axis=0) + \
         lfilter(convolve(-Ao[1,0], Co[0,0]), det, e[:, 0:1], axis=0) + \
         lfilter(convolve(Ao[0,0], Co[1,0]), det, e[:, 1:2], axis=0)

    y = concatenate((y1, y2), axis=1)
    # m = armax(na,nb,nc,nk,u,y)
    m = els(na,nb,nc,nk,u,y)
    t = m.parameters
    t0 = array([])
    for i in range(ny):
        for j in range(ny):
            t0 = concatenate((t0, Ao[i,j][1:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Bo[i,j][nk[i][j]:]))
    for i in range(ny):
            t0 = concatenate((t0, Co[i,0][1:]))
    for i in range(len(t0)):
        print(t0[i],"\t##\t",t[i])

    chiv = get_value_elipse(t,t0,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

test_armax_mimo()

def test_armax_random_mimo():
    n = 5
    ny = 2 #randint(2,n)        #number of outputs (single)
    nu = 2 #randint(2,n)        #number of inputs

    nb =  randint(1,n,(ny,nu)) #ny x nu
    na = randint(1,n,(ny,ny)) #ny x ny
    nc = randint(1,n,(ny,1)) #1 x nu
    nk = 1 + randint(0, n,(ny,nu)) #array([[1,1],[1,1]])
    #nk[i,j] delay of uj input for yi output
    #nb[i,j] order of uj input for yi output

    N = 1000                # Number of samples
    e = 0.01*randn(N, ny)    # Emulates Gaussian white noise with std = 0.01


    #Generate Polynomials
    # Ao = zeros((ny,ny),dtype=object)
    # for i in  range(ny):
    #     for j in range(ny):
    #         Ao[i,j] = gen_stable_poly(na[i,j])
    #         if i != j:
    #             Ao[i,j][0] = 0
    Ao, det = gen_stable_model2x2(na)

    Bo = zeros((ny,nu), dtype=object)
    Co = zeros((ny,1), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(ny):
        for j in range(nu):
            Boo = -1 + 2*randn(nb[i,j]) #generate B's
            Bo[i,j] = concatenate(([0,]*nk[i,j],Boo))
    for i in range(ny):
        Co[i,0] = gen_stable_poly(nc[i,0]+1)

    u = -sqrt(3) + 2*sqrt(3)*rand(N, nu)   # Defines input signal

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = zeros((N,ny),dtype=float) #y's, com ny linhas e N colunas, cada linha é uma saida
    L = max(amax(na),amax(nb+nk),amax(nc)) #to know were to start
    for i in range(L,N):
        for j in range(ny): # for each output
            for k in range(ny): # to travel in cols of the Ao matrix
                # [::-1] makes the array backwards
                y[i,j] += dot(Ao[j,k][1:],-y[i-(len(Ao[j,k])-1):i,k][::-1])
            for k in range(nu):# for each input
                y[i,j] += dot(Bo[j,k][nk[j,k]:],u[i-len(Bo[j,k][nk[j,k]:]):i,k][::-1])
            y[i,j] += dot(Co[j,0][1:],e[i-len(Co[j,0][1:]):i,j][::-1])
    
    t0 = array([])
    for i in range(ny):
        for j in range(ny):
            t0 = concatenate((t0, Ao[i,j][1:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Bo[i,j][nk[i,j]:]))
    for i in range(ny):
        t0 = concatenate((t0, Co[i,0][1:]))

    m = armax(na,nb-ones(nb.shape,dtype=int),nc,nk,u,y)
    t = m.parameters
    # print("Ao: ",Ao)
    # print("A: ",m.A,"\n\n")
    # print(Bo)
    # print(Co)
    # print(na,nb,nc)
    # print(t0.shape,t.shape)
    for i in range(len(t)):
        print(t[i],"\t ## \t", t0[i])
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))


# Test OE
@pytest.fixture
def test_polynomials_oe_siso():
    Ao = [1, -1.2, 0.36]
    Bo = [0, 0.5, 0.1]

    return [Ao, Bo]

@pytest.fixture
def test_signals_oe_siso(test_polynomials_oe_siso):
    # True test system parameters
    Ao = test_polynomials_oe_siso[0]
    Bo = test_polynomials_oe_siso[1]

    # Replicates the following experiment:
    # y(t) = Go(q)*u(t) + Ho(q)*e(t),
    # where u(t) is the system input and e(t) white noise
    N = 1000                              # Number of samples
    u = -sqrt(3) + 2*sqrt(3)*rand(N, 1)   # Defines input signal
    e = 0.01*randn(N, 1)                  # Emulates Gaussian white noise with std = 0.01

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = lfilter(Bo, Ao, u, axis=0) + e

    return [u, y]

def test_oe_siso(test_signals_oe_siso, test_polynomials_oe_siso):
    A = array(test_polynomials_oe_siso[0])
    B = array(test_polynomials_oe_siso[1])
    t0 = array(A[1:].tolist() + B[1:].tolist())

    u = test_signals_oe_siso[0]
    # e = 0.01*randn(1000, 1)
    y = test_signals_oe_siso[1]
    # y = lfilter(B, A, u, axis=0) + lfilter([1], A, e, axis=0)

    nb = len(B) - 2
    nk = 1
    nf = len(A) - 1

    m = oe(nb, nf, nk, u, y)
    t = m.parameters
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

def test_oe_random_siso():
    n = 10
    ny = 1
    nu = 1
    nf = 1 + randint(1, n)  #,(ny,ny))
    nb = 1 + randint(1, n)  # ,(ny,nu))
    nk = 1 + randint(1, n)  #,(ny,nu))
    N = 1000
    e = 0.01*randn(N,)    # Emulates Gaussian white noise with std = 0.01

    Fo = gen_stable_poly(nf+1)
    Boo = -1 + 2*randn(nb)
    Bo = array([0,]*nk + Boo.tolist())

    u = -sqrt(3) + 2*sqrt(3)*rand(N,)

    y = lfilter(Bo, Fo, u, axis=0) + e
    #Estimate the model and get only the parameters
    print(y.shape,u.shape)
    m = oe(nb-1, nf, nk, u, y)
    t = m.parameters

    t0 = array(Bo[nk:].tolist() + Fo[1:].tolist())
    # t0 = array(Fo[1:].tolist() + Bo[nk:].tolist())
    for i in range(len(t0)):
        print(t[i],"\t ## \t", t0[i])
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))

# MISO
def test_oe_miso():
    ny = 1
    nu = 2

    nf = [[2 ,2]]
    nb = [[1, 1]]
    nk = [[1, 1]]

    Fo = array([[[1, -1.2, 0.36],[1, -1.4, 0.49]]])

    Bo = array([[[0, 0.5, 0.1],[0, 0.3,-0.2]]])
    print(Bo[0,1])
    N = 1000
    #Take u as uniform
    u = -1 + 2*rand(N, nu)
    #Generate gaussian white noise with standard deviation 0.1
    e = 0.01*randn(N, ny)
    #Calculate the y through S (OE: G(q) = B(q)/F(q) and H(q) = 1)
    y = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + lfilter(Bo[0,1], Fo[0,1], u[:,1:2], axis=0) + e
    #Estimate the model and get only the parameters
    m = oe(nb, nf, nk, u, y)
    t = m.parameters

    t0 = array([])
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Fo[i,j][1:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Bo[i,j][nk[i][j]:]))
    for i in range(len(t0)):
        print(t0[i],"\t##\t",t[i])

    chiv = get_value_elipse(t,t0,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

def test_oe_random_miso():
    n = 10
    ny = 1
    nu = 1 + randint(1, n)
    nf = 1 + randint(1, n,(ny,nu))  #,(ny,nu))
    nb = 1 + randint(1, n,(ny,nu))  # ,(ny,nu))
    nk = 1 + randint(1, n,(ny,nu))  #,(ny,nu))
    N = 1000
    e = 0.01*randn(N,ny)    # Emulates Gaussian white noise with std = 0.01

    Fo = zeros((ny,nu), dtype=object)
    Bo = zeros((ny,nu), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(nu):
        Boo = -1 + 2*randn(nb[0,i]) #generate B's
        Bo[0,i] = concatenate(([0,]*nk[0,i],Boo))
        u[:,i] = -sqrt(3) + 2*sqrt(3)*rand(N,)

    for i in range(ny):
        for j in range(nu):
            Fo[i,j] = gen_stable_poly(nf[i,j]+1)

    y = e[:,0:1]
    for i in range(nu):
        y += lfilter(Bo[0,i], Fo[0,i], u[:,i:i+1], axis=0)
    # y = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + lfilter(Bo[0,1], Fo[0,1], u[:,1:2], axis=0) + e[:,0:1]
    #Estimate the model and get only the parameters
    m = oe(nb-ones(nb.shape,dtype=int), nf, nk, u, y)
    t = m.parameters

    t0 = array([])

    for i in range(nu):
        t0 = concatenate((t0, Bo[0,i][nk[0,i]:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Fo[i,j][1:]))
    for i in range(len(t0)):
        print(t[i],"\t##\t",t0[i])

    chiv = get_value_elipse(t,t0,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

# SIMO

def test_oe_simo():
    ny = 2
    nu = 1

    nf = [[2],[2]]
    nb = [[1],[1]]
    nk = [[1],[1]]

    Fo = array([[[1, -1.2, 0.36]],
                [[1, -1.4, 0.49]]])

    Bo = array([[[0, 0.5, 0.1]],
                [[0, 0.3,-0.2]]])


    N = 1000
    #Take u as uniform
    u = -1 + 2*rand(N, nu)
    #Generate gaussian white noise with standard deviation 0.1
    e = 0.01*randn(N, ny)
    #Calculate the y through S (OE: G(q) = B(q)/F(q) and H(q) = 1)
    y1 = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + e[:,0:1]
    y2 = lfilter(Bo[1,0], Fo[1,0], u[:,0:1], axis=0) + e[:,1:2]
    y = concatenate((y1, y2), axis=1)
    #Estimate the model and get only the parameters
    m = oe(nb, nf, nk, u, y)
    t = m.parameters

    t0 = array([])
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Fo[i,j][1:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Bo[i,j][nk[i][j]:]))
    for i in range(len(t0)):
        print(t0[i],"\t##\t",t[i])

    chiv = get_value_elipse(t,t0,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

def test_oe_random_simo():
    n = 5
    ny = 1 + randint(1, n)
    nu = 1
    nf = 1 + randint(1, n,(ny,nu))  #,(ny,nu))
    nb = 1 + randint(1, n,(ny,nu))  # ,(ny,nu))
    nk = 1 + randint(1, n,(ny,nu))  #,(ny,nu))
    N = 1000
    e = 0.01*randn(N,ny)    # Emulates Gaussian white noise with std = 0.01

    Fo = zeros((ny,nu), dtype=object)
    Bo = zeros((ny,nu), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(ny):
        for j in range(nu):
            Boo = -1 + 2*randn(nb[i,j]) #generate B's
            Bo[i,j] = concatenate(([0,]*nk[i,j],Boo))
            u[:,j] = -sqrt(3) + 2*sqrt(3)*rand(N,)

    for i in range(ny):
        for j in range(nu):
            Fo[i,j] = gen_stable_poly(nf[i,j]+1)

    yt = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + e[:,0:0+1]
    y = yt
    for i in range(1,ny):
        yt = lfilter(Bo[i,0], Fo[i,0], u[:,0:1], axis=0) + e[:,i:i+1]
        y  = concatenate((y, yt), axis=1)

    #Estimate the model and get only the parameters

    m = oe(nb-ones(nb.shape,dtype=int), nf, nk, u, y)
    t = m.parameters


    t0 = array([])

    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Bo[i,j][nk[i,j]:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Fo[i,j][1:]))
    for i in range(len(t0)):
        print(t[i],"\t##\t",t0[i])

    chiv = get_value_elipse(t,t0,inv(m.P))
    assert check_inside_elipse(chiv,len(t))


# MIMO

def test_oe_mimo():
    ny = 2
    nu = 2

    nf = [[2, 2], [2, 2]]
    nb = [[1, 1], [1, 1]]
    nk = [[1, 1], [1, 1]]

    Fo = array([[[1, -1.2, 0.36], [1, -1.6, 0.84]],
                [[1, -1.0, 0.25], [1, -1.4, 0.49]]])
    Bo = array([[[0, 0.5, 0.1], [0, 0.8,-0.4]],
                [[0, 0.7, 0.2], [0, 0.3,-0.2]]])

    N = 1000
    #Take u as uniform
    u = -1 + 2*rand(N, nu)

    e = 0.01*randn(N, ny)

    y1 = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + lfilter(Bo[0,1], Fo[0,1], u[:,1:2], axis=0) + e[:,0:1]
    y2 = lfilter(Bo[1,0], Fo[1,0], u[:,0:1], axis=0) + lfilter(Bo[1,1], Fo[1,1], u[:,1:2], axis=0) + e[:,1:2]
    y = concatenate((y1, y2), axis=1)

    m = oe(nb, nf, nk, u, y)
    t = m.parameters

    to = array([])
    for i in range(ny):
        for j in range(nu):
            to = concatenate((to, Fo[i,j][1:]))
    for i in range(ny):
        for j in range(nu):
            to = concatenate((to, Bo[i,j][nk[i][j]:]))
    for i in range(len(to)):
        print(to[i],"\t##\t",t[i])

    chiv = get_value_elipse(t,to,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

def test_oe_random_mimo():
    n = 5
    ny = 1 + randint(1, n)
    nu = 1 + randint(1, n)
    nf = 1 + randint(1, n,(ny,nu))  #,(ny,nu))
    nb = 1 + randint(1, n,(ny,nu))  # ,(ny,nu))
    nk = 1 + randint(1, n,(ny,nu))  #,(ny,nu))
    N = 1000
    e = 0.01*randn(N,ny)    # Emulates Gaussian white noise with std = 0.01

    Fo = zeros((ny,nu), dtype=object)
    Bo = zeros((ny,nu), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(ny):
        for j in range(nu):
            Boo = -1 + 2*randn(nb[i,j]) #generate B's
            Bo[i,j] = concatenate(([0,]*nk[i,j],Boo))
            u[:,j] = -sqrt(3) + 2*sqrt(3)*rand(N,)

    for i in range(ny):
        for j in range(nu):
            Fo[i,j] = gen_stable_poly(nf[i,j]+1)


    yt = e[:,0:1]
    for j in range(nu):
        yt += lfilter(Bo[0,j], Fo[0,j], u[:,j:j+1], axis=0)
    y = yt
    for i in range(1,ny):
        yt = e[:,i:i+1]
        for j in range(nu):
            yt += lfilter(Bo[i,j], Fo[i,j], u[:,j:j+1], axis=0)
        y = concatenate((y, yt),axis = 1)
    # y1 = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + lfilter(Bo[0,1], Fo[0,1], u[:,1:2], axis=0) + e[:,0:1]
    # y2 = lfilter(Bo[1,0], Fo[1,0], u[:,0:1], axis=0) + lfilter(Bo[1,1], Fo[1,1], u[:,1:2], axis=0) + e[:,1:2]
    # y = concatenate((y1, y2), axis=1)
    #Estimate the model and get only the parameters
    m = oe(nb-ones(nb.shape,dtype=int), nf, nk, u, y)
    t = m.parameters


    t0 = array([])

    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Bo[i,j][nk[i,j]:]))
    for i in range(ny):
        for j in range(nu):
            t0 = concatenate((t0, Fo[i,j][1:]))
    for i in range(len(t0)):
        print(t[i],"\t##\t",t0[i])

    chiv = get_value_elipse(t,t0,inv(m.P))
    assert check_inside_elipse(chiv,len(t))


# -----------------Recursive Module-----------------
def test_rls_siso(test_signals_arx_siso,test_polynomials_arx_siso):
    A = array(test_polynomials_arx_siso[0])
    B = array(test_polynomials_arx_siso[1])
    t0 = array(A[1:].tolist() + B[1:].tolist())

    u = test_signals_arx_siso[0]
    y = test_signals_arx_siso[1]
    nk = 1

    m = rls(len(A)-1, len(B)-(nk+1), nk , u, y)
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
    ny = 1
    nu = 1

    nf = 2
    nb = 1
    nc = 2
    nd = 2
    nk = 1

    Fo  = [1, -1.2, 0.36]
    Bo = [0, 0.5, 0.1]
    Co = [1, 0.8, 0.2]
    Do = [1, -1.6, 0.64]

    N = 1000
    #Take u as uniform
    u = -1 + 2*randn(N, nu)

    e = 0.1*randn(N, ny)

    y = lfilter(Bo, Fo, u, axis=0) + lfilter(Co, Do, e, axis=0)
    m = bj(nb, nc, nd, nf, nk, u, y)
    t = m.parameters

    to = array([])
    to = concatenate((to, Fo[1:]))
    to = concatenate((to, Bo[1:]))
    to = concatenate((to, Co[1:]))
    to = concatenate((to, Do[1:]))

    chiv = get_value_elipse(t,to,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

def test_bj_random_siso():
    n = 2
    ny = 1
    nu = 1
    nf = 2#1 + randint(1, n)  #,(ny,ny))
    nb = 2 #+ randint(1, n)  # ,(ny,nu))
    nc = 2#1 + randint(1, n)
    nk = 1# + randint(1, n)  #,(ny,nu))
    nd = 2#1 + randint(1, n)

    N = 1000
    e = 0.01*randn(N,)    # Emulates Gaussian white noise with std = 0.01

    Fo = gen_stable_poly(nf+1).tolist()
    Co = gen_stable_poly(nc+1).tolist()
    Do = gen_stable_poly(nd+1).tolist()
    Boo = -1 + 2*randn(nb)
    Bo = array([0,]*nk + Boo.tolist()).tolist()
    # Fo  = [1, -1.2, 0.36]
    # Bo = [0, 0.5, 0.1]
    # Co = [1, 0.8, 0.2]
    # Do = [1, -1.6, 0.64]
    u = -sqrt(3) + 2*sqrt(3)*rand(N,)

    y = lfilter(Bo, Fo, u, axis=0) + lfilter(Co, Do, e, axis=0)
    #Estimate the model and get only the parameters
    print(Fo,Bo,Co,Do)
    print(nf,nb,nc,nd)
    m = bj(nb-1, nc, nd, nf, nk, u, y)
    t = m.parameters
    #ValueError: Residuals are not finite in the initial point
    t0 = array(Bo[nk:] + Co[1:] + Do[1:] + Fo[1:])
    for i in range(len(t0)):
        print(t[i],"\t ## \t", t0[i])
    chivalue = get_value_elipse(t, t0, inv(m.P))
    assert check_inside_elipse(chivalue, len(t))


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

    N = 1000
    # Todo: parametrize in terms of ny, nu, nb, nc, nd, nf, nk 
    u = -sqrt(3) + 2*sqrt(3)*randn(N, 1)
    e = 0.01*randn(N, 2)
    y1 = lfilter(B1o, F1o, u[:,0:1], axis=0) + lfilter(C1o, D1o, e[:,0:1], axis=0)
    y2 = lfilter(B2o, F2o, u[:,0:1], axis=0) + lfilter(C2o, D2o, e[:,1:2], axis=0)
    y = concatenate((y1, y2), axis=1)

    return [u,y]

def test_bj_simo():
    ny = 2
    nu = 1

    Fo = array([[[1, -1.2, 0.36]],[[1, -1.4, 0.49]]])
    Bo = array([[[0, 0.5, 0.1]],[[0, 0.3,-0.2]]])
    Co = array([[[1, 0.8, 0.16]],[[1, 0.9, 0.22]]])
    Do = array([[[1, -1.8, 0.91]],[[1, -1.6, 0.8]]])

    nf = [[2], [2]]
    nb = [[1], [1]]
    nc = [[2], [2]]
    nd = [[2], [2]]
    nk = [[1], [1]]


    N = 1000
    e = 0.01*randn(N, ny)
    u = -1 + 2*randn(N, nu)

    y1 = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + lfilter(Co[0,0], Do[0,0], e[:,0:1], axis=0)
    y2 = lfilter(Bo[1,0], Fo[1,0], u[:,0:1], axis=0) + lfilter(Co[1,0], Do[1,0], e[:,1:2], axis=0)
    y = concatenate((y1, y2), axis=1)

    m = bj(nb, nc, nd, nf, nk, u, y)
    t = m.parameters

    to = array([])
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Fo[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Bo[iy,iu][nk[iy][iu]:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Co[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Do[iy,iu][1:]))

    chiv = get_value_elipse(t,to,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

def test_bj_random_simo():
    n = 2
    ny = 2
    nu = 1
    nf = 1 + randint(1, n ,(ny,nu))
    nb = 1 + randint(1, n ,(ny,nu))
    nc = 1 + randint(1, n ,(ny,nu))
    nk = 1 + randint(1, n,(ny,nu))
    nd = 1 + randint(1, n ,(ny,nu))

    N = 1000
    e = 0.01*randn(N,ny)    # Emulates Gaussian white noise with std = 0.01

    Fo = zeros((ny,nu), dtype=object)
    Bo = zeros((ny,nu), dtype=object)
    Co = zeros((ny,nu), dtype=object)
    Do = zeros((ny,nu), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(ny):
        for j in range(nu):
            Boo = -1 + 2*randn(nb[i,j]) #generate B's
            Bo[i,j] = concatenate(([0,]*nk[i,j],Boo))
            u[:,j] = -sqrt(3) + 2*sqrt(3)*rand(N,)

    for i in range(ny):
        for j in range(nu):
            Fo[i,j] = gen_stable_poly(nf[i,j]+1)
            Co[i,j] = gen_stable_poly(nc[i,j]+1)
            Do[i,j] = gen_stable_poly(nd[i,j]+1)


    # Fo =  gen_stable_poly(nf+1).tolist()
    # Co = gen_stable_poly(nc+1).tolist()
    # Do = gen_stable_poly(nd+1).tolist()
    # Boo = -1 + 2*randn(nb)
    # Bo = array([0,]*nk + Boo.tolist()).tolist()

    print(Bo[0,0])
    print(Fo[0,0])
    print(Co[0,0])
    print(Do[0,0])
    y1 = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + lfilter(Co[0,0], Do[0,0], e[:,0:1], axis=0)
    y2 = lfilter(Bo[1,0], Fo[1,0], u[:,0:1], axis=0) + lfilter(Co[1,0], Do[1,0], e[:,1:2], axis=0)
    y = concatenate((y1, y2), axis=1)

    m = bj(nb-ones(nb.shape,dtype=int), nc, nd, nf, nk, u, y)
    t = m.parameters

    to = array([])
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Bo[iy,iu][nk[iy][iu]:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Co[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Do[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Fo[iy,iu][1:]))

    chiv = get_value_elipse(t,to,inv(m.P))
    assert check_inside_elipse(chiv,len(t))
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

    N = 1000
    u = -sqrt(3) + 2*sqrt(3)*randn(N, int((len(test_polynomials_bj_miso) - 2)/2))
    e = 0.01*randn(N, 1)
    y1 = lfilter(B1o, F1o, u[:,0:1], axis=0)
    y2 = lfilter(B2o, F2o, u[:,1:2], axis=0)
    ye = lfilter(Co, Do, e[:,0:1], axis=0)
    y = y1 + y2 + ye

    return [u,y]

def test_bj_miso():
    ny = 1
    nu = 2

    Fo = array([[[1, -1.2, 0.36],[1, -1.4, 0.49]]])
    Bo = array([[[0, 0.5, 0.1],[0, 0.3,-0.2]]])
    Co = array([[[1, 0.8, 0.16]]])
    Do = array([[[1, -1.8, 0.91]]])

    nf = [[2, 2]]
    nb = [[1, 1]]
    nc = [[2, 2]]
    nd = [[2, 2]]
    nk = [[1, 1]]

    N = 1000
    #Take u as uniform
    u = -1 + 2*randn(N, nu)
    e = 0.01*randn(N, ny)

    y = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + \
        lfilter(Bo[0,1], Fo[0,1], u[:,1:2], axis=0) + \
        lfilter(Co[0,0], Do[0,0], e[:,0:1], axis=0)

    m = bj(nb, nc, nd, nf, nk, u, y)
    t = m.parameters

    to = array([])
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Fo[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Bo[iy,iu][nk[iy][iu]:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Co[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Do[iy,iu][1:]))

    chiv = get_value_elipse(t,to,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

def test_bj_random_miso():
    n = 2
    ny = 1
    nu = 2
    nf = 1 + randint(1, n ,(ny,nu))
    nb = 1 + randint(1, n ,(ny,nu))
    nc = 1 + randint(1, n ,(ny,nu))
    nk = 1 + randint(1, n,(ny,nu))
    nd = 1 + randint(1, n ,(ny,nu))

    N = 1000
    e = 0.01*randn(N,ny)    # Emulates Gaussian white noise with std = 0.01

    Fo = zeros((ny,nu), dtype=object)
    Bo = zeros((ny,nu), dtype=object)
    Co = zeros((ny,nu), dtype=object)
    Do = zeros((ny,nu), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(ny):
        for j in range(nu):
            Boo = -1 + 2*randn(nb[i,j]) #generate B's
            Bo[i,j] = concatenate(([0,]*nk[i,j],Boo))
            u[:,j] = -sqrt(3) + 2*sqrt(3)*rand(N,)

    for i in range(ny):
        for j in range(nu):
            Fo[i,j] = gen_stable_poly(nf[i,j]+1)
            Co[i,j] = gen_stable_poly(nc[i,j]+1)
            Do[i,j] = gen_stable_poly(nd[i,j]+1)


    y = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + \
        lfilter(Bo[0,1], Fo[0,1], u[:,1:2], axis=0) + \
        lfilter(Co[0,0], Do[0,0], e[:,0:1], axis=0)

    m = bj(nb-ones(nb.shape,dtype=int), nc, nd, nf, nk, u, y)
    t = m.parameters

    to = array([])
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Bo[iy,iu][nk[iy][iu]:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Co[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Do[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Fo[iy,iu][1:]))

    chiv = get_value_elipse(t,to,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

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

def test_bj_mimo():
    ny = 2
    nu = 2

    Fo = array([[[1, -1.2, 0.36], [1, -1.8, 0.91]],
                [[1, -1.4, 0.49], [1, -1.0, 0.25]]])
    Bo = array([[[0, 0.5, 0.1], [0, 0.9,-0.1]],
                [[0, 0.4, 0.8], [0, 0.3,-0.2]]])
    Co = array([[[1, 0.8, 0.16]],[[1, 0.9, 0.22]]])
    Do = array([[[1, -1.8, 0.91]],[[1, -1.6, 0.8]]])

    nf = [[2, 2], [2, 2]]
    nb = [[1, 1], [1, 1]]
    nc = [[2], [2]]
    nd = [[2], [2]]
    nk = [[1, 1], [1, 1]]

    N = 1000
    #Take u as uniform
    u = -1 + 2*randn(N, nu)
    e = 0.01*randn(N, ny)

    y1 = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + \
        lfilter(Bo[0,1], Fo[0,1], u[:,1:2], axis=0) + lfilter(Co[0,0], Do[0,0], e[:,0:1], axis=0)
    y2 = lfilter(Bo[1,0], Fo[1,0], u[:,0:1], axis=0) + \
        lfilter(Bo[1,1], Fo[1,1], u[:,1:2], axis=0) + lfilter(Co[1,0], Do[1,0], e[:,1:2], axis=0)
    y = concatenate((y1, y2), axis=1)

    m = bj(nb, nc, nd, nf, nk, u, y)
    t = m.parameters

    to = array([])
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Fo[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Bo[iy,iu][nk[iy][iu]:]))
    for iy in range(ny):
        for iu in range(1):
            to = concatenate((to, Co[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(1):
            to = concatenate((to, Do[iy,iu][1:]))

    chiv = get_value_elipse(t,to,inv(m.P))
    assert check_inside_elipse(chiv,len(t))

def test_bj_random_mimo():
    n = 2
    ny = 2
    nu = 2
    nf = 1 + randint(1, n ,(ny,nu))
    nb = 1 + randint(1, n ,(ny,nu))
    nc = 1 + randint(1, n ,(ny,nu))
    nk = 1 + randint(1, n,(ny,nu))
    nd = 1 + randint(1, n ,(ny,nu))

    N = 1000
    e = 0.01*randn(N,ny)    # Emulates Gaussian white noise with std = 0.01

    Fo = zeros((ny,nu), dtype=object)
    Bo = zeros((ny,nu), dtype=object)
    Co = zeros((ny,nu), dtype=object)
    Do = zeros((ny,nu), dtype=object)
    u = zeros((N,nu), dtype=float)

    #Generates Bo's and the inputs
    for i in range(ny):
        for j in range(nu):
            Boo = -1 + 2*randn(nb[i,j]) #generate B's
            Bo[i,j] = concatenate(([0,]*nk[i,j],Boo))
            u[:,j] = -sqrt(3) + 2*sqrt(3)*rand(N,)

    for i in range(ny):
        for j in range(nu):
            Fo[i,j] = gen_stable_poly(nf[i,j]+1)
            Co[i,j] = gen_stable_poly(nc[i,j]+1)
            Do[i,j] = gen_stable_poly(nd[i,j]+1)


    y1 = lfilter(Bo[0,0], Fo[0,0], u[:,0:1], axis=0) + \
        lfilter(Bo[0,1], Fo[0,1], u[:,1:2], axis=0) + lfilter(Co[0,0], Do[0,0], e[:,0:1], axis=0)
    y2 = lfilter(Bo[1,0], Fo[1,0], u[:,0:1], axis=0) + \
        lfilter(Bo[1,1], Fo[1,1], u[:,1:2], axis=0) + lfilter(Co[1,0], Do[1,0], e[:,1:2], axis=0)
    y = concatenate((y1, y2), axis=1)

    m = bj(nb-ones(nb.shape,dtype=int), nc, nd, nf, nk, u, y)
    t = m.parameters

    to = array([])
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Bo[iy,iu][nk[iy][iu]:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Co[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Do[iy,iu][1:]))
    for iy in range(ny):
        for iu in range(nu):
            to = concatenate((to, Fo[iy,iu][1:]))

    chiv = get_value_elipse(t,to,inv(m.P))
    assert check_inside_elipse(chiv,len(t))
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