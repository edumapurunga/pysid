# -*- coding: utf-8 -*-

"""
Module for recursive methods
"""

from numpy import zeros, identity, matmul, empty, insert, concatenate, power, \
    array, amax, eye, shape, kron, sum, sqrt, dot, asmatrix
from numpy.random import rand, randn
from math import ceil
from pysid.identification.models import polymodel
from pysid.identification.solvers import qrsol,qrsoliv
from pysid.io.check import chckin
from pysid.identification.pemethod import filtmat, arx
#from pysid.identification.ivmethod import iv2
from scipy.linalg import inv, toeplitz
from scipy.signal import lfilter     #To generate the data
import warnings

__all__ = ['els', 'rls']

def var_erro(error, th = 0.001):
    """
    
    Measures if the error between predictions is,percentage, less than th 
    
    internal use
    Parameters
    ----------
    error : numpy array
    th : float
        DESCRIPTION. The default is 0.001.

    Returns
    -------
    bool.

    """
    #If the error has varied less than th between iterations, it returns False and exits the while
    if error[0,0] != 0 :
        for i in range(error.shape[1]):
            if (abs((error[0,i] - error[1,i])/error[0,i])) > th :
                return True
        return False
    else:
        return True

def els(na,nb,nc,nk,u,y,th = 0.001,n_max = 150):
    """
    
    Performs the Extended Least Squres algorithm on u,y data,
    indentifing A,B and C polynomials with na,nb and nc degree respectively.
    
    Parameters
    ----------
    na : int
        Degree of A polynomial
    nb : int
        Degree of B polynomial
    nc : int
        Degree of C polynomial
    nk : int
        minimum delay for B polynomial
    u : numpy array
        Array (or array of arrays) contaning the inputs chronologically
    y : numpy array
        Array (or array of arrays) contaning the outputs chronologically
    th : float
        Treshold
        Minimum difference between the quadratic sum of two consecutive errors needed to assume that there was convergence
    n_max : int
        Maximum number of iterations allowed
        
    Returns
    -------
    m : pysid polymodel

    """
    na, nb, nc, _, _, nk, u, y = chckin(na, nb, nc, [], [], nk, u, y)
    Nu, nu = u.shape
    Ny, ny = y.shape

    nbk = nb+nk
    w = max(amax(na),amax(nbk),amax(nc))
    da = sum(sum(na))
    db = sum(sum(nb+1))
    dc = sum(sum(nc))

    y_sol = y[w:,:] #output used for theta in solving the linear problem
    psi = zeros((ny,1), dtype = object) #one line for each output
    u_cont = 0
    y_cont = 0
    # nb number of prior samples that will be looked
    # nk were I start to look(-1, -2, -3,...)
    for i in range(ny):
        u_reg = zeros([Nu-w,int(sum(nb[i,:]+1))])
        y_reg = zeros([Ny-w,int(sum(na[i,:]))])
        for j in range(nu):
            for inb in range(nb[i,j]+1):
                u_reg[:,u_cont] = u[w-nbk[i,j]+inb:Nu-nbk[i,j]+inb,j]
                u_cont = u_cont + 1
        for jj in range(ny):
            for ina in range(na[i,jj]):
                y_reg[:,y_cont] = -y[w-na[i,jj]+ina:Ny-na[i,jj]+ina,jj]
                y_cont = y_cont + 1
        u_cont = 0
        y_cont = 0
        psi[i,0] = concatenate((u_reg,y_reg),axis=1)

    theta_mq = zeros((1,ny),dtype=object)
    #Standard LS : theta = pseudo_inverse(psi) * y
    for i in range(ny):
        theta_mq[0,i] = qrsol(psi[i,0],y_sol[:,i].reshape((y_sol.shape[0],1)))[0]
    # theta_mq = inv(psi[0,0].T @ psi[0,0]) @ psi[0,0].T @ y_sol[:,0]

    #make psi_emq (first time)
    psi_emq = zeros(psi.shape,dtype=object)
    res = zeros((ny,1),dtype=object)
    theta_emq = zeros((1,ny),dtype=object)
    #prepares psi_emq and calculate the residuals
    for i in range(ny):
        psi_emq[i,0] = zeros((y_sol.shape[0]-nc[i,0],psi[i,0].shape[1]+nc[i,0]))
        res[i,0] = y_sol[:,i] - matmul(psi[i,0],theta_mq[0,i])

    #Now each output has its own regressor, since the residuals are different.
    for i in range(ny):
        psi_emq[i,0][:,:psi[i,0].shape[1]] = psi[i,0][nc[i][0]:,:]
        for j in range(nc[i][0]):
            res_i = res[i,0][j:-nc[i][0]+j]
            psi_emq[i,0][:,psi_emq[i,0].shape[1]-nc[i][0]+j] = res_i
        #first theta_emq, setting its shape
        # theta_emq[0,i] = zeros((psi_emq[i,0].shape[1],))

    #it is necessary to get the right theta, which was calculated with the residue of the corresponding output
    for i in range(ny):
        theta_emq[0,i] = qrsol(psi_emq[i,0][:,:],y_sol[nc[i][0]:,i].reshape(y_sol[nc[i][0]:,i].shape[0],1))[0]

    #Iteractions
    s_error = zeros((2,ny)) #one line for past error and one for current error
    n_iteracoes = n_max
    i = 0
    while (var_erro(s_error) and i<n_iteracoes):
        for j in range(ny): #for each psi
            res[j,0] = y_sol[nc[j][0]:,j] - matmul(psi_emq[j,0],theta_emq[0,j])#each j has the residual to go in a regressor
            for k in range(nc[j][0]): #addition of residue in psi_emq
                # res_i = res[j,0][k:-nc[j][0]+k]
                res_i = res[j,0]
                n_zeros = nc[j][0]-k
                res_i = concatenate(([0,]*n_zeros, res_i[:-1*n_zeros]))
                # res_i[:-nc[j][0]+k] = res[nc[j][0]-k:,j]
                psi_emq[j,0][:,-nc[j][0]+k] = res_i
            theta_emq[0,j] = qrsol(psi_emq[j,0][:,:],y_sol[nc[j][0]:,j].reshape(y_sol[nc[j][0]:,j].shape[0],1))[0]

        s_error[1,:] = s_error[0,:]
        for ii in range(ny):
            s_error[0,ii] = sum(power(res[ii,0],2))

        i = i + 1

    #Assembly of matrices of polynomials
    A = empty((ny,ny), dtype='object')
    B = empty((ny,nu), dtype='object')
    C = empty((ny,1),  dtype='object')

    nbp = nb+1

    #Fill each element of B with zeros referring to nk
    # If nk = 2, B must be [0 0 bo b1 b2]

    for i in range(ny):
        for j in range(nu):
            if j == 0:
                B[i,j] = theta_emq[0,i][0:nbp[i,j]][::-1]
            else:
                B[i,j] = theta_emq[0,i][sum(nbp[i,0:j]):sum(nbp[i,0:j])+nbp[i,j]][::-1]
            B[i,j] = insert(B[i,j],0,zeros((nk[i,j]))) #fill with zeros

    for i in range(ny):
        for j in range(ny):
            if j==0:
                A[i,j] = theta_emq[0,i][sum(nbp[i,:]):sum(nbp[i,:])+na[i,j]][::-1]
            else:
                A[i,j] = theta_emq[0,i][sum(nbp[i,:])+sum(na[i,0:j]):sum(nbp[i,:])+sum(na[i,0:j])+na[i,j]][::-1]
            if i==j:
                A[i,j] = insert(A[i,j],0,1)
            else:
                A[i,j] = insert(A[i,j],0,0)

    for c in range(ny):
        C[c,0] = theta_emq[0,c][::-1][0:nc[c,0]]
        C[c,0] = insert(C[c,0],0,1)

    #create de object 
    n_params = theta_emq.shape[0]*theta_emq.shape[1]
    m = polymodel('armax',A,B,C,None,None,nk,n_params+1,(u,y),nu,ny,1)
    par = array([])
    for i in range(ny):
        for j in range(ny):
            par = concatenate((par, A[i,j][1:]))
    for i in range(ny):
        for j in range(nu):
            par = concatenate((par, B[i,j][nk[i][j]:]))
    for i in range(ny):
            par = concatenate((par, C[i,0][1:]))
    m.setparameters(par)

    ehat = (filtmat(A, y, C) - filtmat(B, u, C))
    # Get covariance of ehat
    sig = (ehat.T @ ehat)/Ny
    # Inverse of sig
    isig = inv(sig)
    M = zeros((da + db + dc, da + db + dc))
    psi_emq_total = array([])
    if ny > 1:
        # psi_emq_total = add(psi_emq_total,psi_emq[0,0],out=psi_emq_total,casting="unsafe")
        psi_emq_total = psi_emq[0,0]
        for i in range(1,ny):
            psi_emq_total = concatenate((psi_emq_total, psi_emq[i,0]),axis=1)
    else:
        psi_emq_total = psi_emq[0,0]
    for k in range(0, psi_emq_total.shape[0], ny):
        M += psi_emq_total[k:k+ny, :].T @ isig @ psi_emq_total[k:k+ny, :]
    M /= Ny
    m.M = M
    m.setcov(sig**2, inv(M)/Ny, sig)
    return m

def rls(na,nb,nk,u,y):
    """
    Performs the Recursive Least Squres algorithm on u,y data,
    indentifing A,B and C polynomials with na,nb and nc degree respectively.

    **Only implemented for SISO cases

    Parameters
    ----------
    na : int
        Degree of A polynomial
    nb : int
        Degree of B polynomial
    nk : int
        minimum delay for B polynomial
    u : numpy array
        Array (or array of arrays) contaning the inputs chronologically
    y : numpy array
        Array (or array of arrays) contaning the outputs chronologically

    Returns
    -------
    m : pysid polymodel

    """
    ny = y.shape[1]
    nu = u.shape[1]

    n_params = na+(nb+1)
    w = max(na,nb+nk)
    nb = nb+1 

    theta = zeros([n_params,ny])
    P = identity(n_params)*1e5 # mimo case will have a pk for each output

    # first data I put manually to avoid an if in the for
    u=u[w:,:]
    y=y[w:,:]

    cont = w
    psik = zeros((1,n_params))
    for i in range(y.shape[0]-w):
        for j in range(nu): 
            psik[0,:nb-1] = psik[0,1:nb]
        psik[0,nb-1] = u[cont-nk]
        for j in range(ny):
            psik[0,nb:nb+na-1] =  psik[0,nb+1:]
        psik[0,nb+na-1] = -y[cont-1]
        K = matmul(P,psik.transpose())/((matmul( psik,matmul(P,psik.transpose()) ) + 1)[0,0])
        theta = theta + K*(y[cont]-matmul(psik,theta))
        P = P - K*(matmul(psik,P))
        cont = cont + 1

    #Assembly of matrices of polynomials 
    A = empty((ny,ny), dtype='object')
    B = empty((ny,nu), dtype='object')

    nbp = nb+1
    #Fill each element of B with zeros referring to nk
    # If nk = 3, B must be [0 0 bo b1 b2]

    for c in range(ny):
        for i in range(nu):
            if i == 0:
                B[c,i] = theta[nbp-2::-1,c]
            else:
                B[c,i] = theta[i*(nbp-1)+nbp:(nbp-1)*i:-1,c]
            B[c,i] = insert(B[c,i],0,zeros((nk))) #fill with zeros

    for c in range(ny):
        for i in range(nu):
            if i == 0:
                A[c,i] = theta[(nbp*nu)+i*(na)+na-1:(nbp*nu)-2:-1,c]
            else:
                A[c,i] = theta[(nbp*nu)+i*(na-1)+na:(nbp*nu)+(na-1)*i:-1,c]
            if i==c:
                A[c,i] = insert(A[c,i],0,1)
            else:
                A[c,i] = insert(A[c,i],0,0)

    #Create the object
    n_params = theta.shape[0]*theta.shape[1]
    m = polymodel('arx',A,B,None,None,None,nk,n_params+1,(u,y),nu,ny,1)
    p = theta.T.tolist()[0]
    p.reverse()
    m.setparameters(p)
    m.setcov(None, P, None)
    return m
