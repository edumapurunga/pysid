# -*- coding: utf-8 -*-

"""
Module for recursive methods
"""

from numpy import zeros, identity, matmul, empty, insert, concatenate, power
from .models import polymodel
from .solvers import qrsolm

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

def els(na,nb,nc,nk,u,y,th = 0.001,n_max = 100):
    """
    
    Performs the Extended Least Squres algorithm on u,y data,
    indentifing A,B and C polynomials with na,nb and nc degree respectively.
    
    **The version takes presuposses that the order of every polynomial (in not siso cases)
    is the same. So A11 has the same order as any other Aij , same goes for B11 with any Bij
    
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
    nu = u.shape[1]
    ny = y.shape[1]

    nbk = nb+nk

    w = max(na,nbk,nc)

    y_sol = y[w:] #output used for theta in solving the linear problem
    u_reg = zeros([u.shape[0]-w,nu*(nb+1)])
    cont = 0
    # nb number of prior samples that will be looked
    # nk were I start to look(-1, -2, -3,...)
    for c in range(nu): #for each input
        for i in range((nb+1)): #u on psi goes from u[k-nk] to u[k-nb-nk+1]
            u_reg[:,cont] = u[w-nbk+i:u.shape[0]-nbk+i,c]
            cont = cont + 1
    cont = 0
    y_reg = zeros([y.shape[0]-w,ny*na])
    for c in range(ny):
        for i in range(na):
            y_reg[:,cont] = -y[w-na+i:y.shape[0]-na+i,c]
            cont = cont + 1 
    psi = concatenate((u_reg,y_reg),axis=1)

    #Standard LS : theta = pseudo_inverse(psi) * y
    theta_mq = qrsolm(psi,y_sol)

    #make psi_emq (first time)

    psi_emq = zeros((y_sol.shape[0]-nc,psi.shape[1]+nc,ny))
    res = y_sol - matmul(psi,theta_mq)
    #Agora cada saida tem seu próprio regressor, já que os residuos são diferentes
    #Now each output has its own regressor, since the residuals are different.
    for i in range(ny):
        psi_emq[:,:psi.shape[1],i] = psi[nc:,:]
        for j in range(nc):
            res_i = res[j:-nc+j,i]
            psi_emq[:,psi_emq.shape[1]-nc+j,i] = res_i


    #first theta_emq
    theta_emq = zeros((psi_emq.shape[1],ny))
    
    #it is necessary to get the right theta, which was calculated with the residue of the corresponding output
    for i in range(ny):
        theta_emq[:,i] = qrsolm(psi_emq[:,:,i],y_sol[nc:,:])[:,i]

    #Iteractions
    res = zeros((psi_emq.shape[0],nc))
    res_i = zeros((res.shape[0]))
    s_error = zeros((2,ny)) #one line for past error and one for current error
    n_iteracoes = n_max
    i = 0
    while (var_erro(s_error) and i<n_iteracoes):
        for j in range(ny): #for each psi
            res = (y_sol[nc:] - matmul(psi_emq[:,:,j],theta_emq[:,:])) #each j has the residual to go in a regressor
            for n in range(nc): #addition of residue in psi_emq
                res_i[:-nc+n] = res[nc-n:,j]
                psi_emq[:,-nc+n,j] = res_i
            theta_emq[:,j] = qrsolm(psi_emq[:,:,j],y_sol[nc:,:])[:,j]

        s_error[1,:] = s_error[0,:]
        for ii in range(ny):
            s_error[0,ii] = sum(power(res,2)[:,ii])

        i = i + 1

    #Assembly of matrices of polynomials
    A = empty((ny,ny), dtype='object')
    B = empty((ny,nu), dtype='object')
    C = empty((ny,1),  dtype='object')

    nbp = nb+1

    #Fill each element of B with zeros referring to nk
    # If nk = 3, B must be [0 0 bo b1 b2]

    for c in range(ny):
        for i in range(nu):
            if i == 0:
                B[c,i] = theta_emq[i*(nbp)-1+nbp::-1,c]
            else:
                B[c,i] = theta_emq[i*(nbp-1)+nbp:(nbp-1)*i:-1,c]
            B[c,i] = insert(B[c,i],0,zeros((nk))) #fill with zeros

    for c in range(ny):
        for i in range(nu):
            if i == 0:
                A[c,i] = theta_emq[(nbp*nu)+i*(na)+na-1:(nbp*nu)-1:-1,c]
            else:
                A[c,i] = theta_emq[(nbp*nu)+i*(na-1)+na:(nbp*nu)+(na-1)*i:-1,c]
            if i==c:
                A[c,i] = insert(A[c,i],0,1)
            else:
                A[c,i] = insert(A[c,i],0,0)

    for c in range(ny):
        C[c,0] = theta_emq[:(nbp*nu)+(na*ny)-1:-1,c].reshape((1,nc))
        C[c,0] = insert(C[c,0],0,1)

    #create de object 
    n_params = theta_emq.shape[0]*theta_emq.shape[1]
    m = polymodel('armax',A,B,C,None,None,nk,n_params+1,(u,y),nu,ny,1)

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
    return m