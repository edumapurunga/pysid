# -*- coding: utf-8 -*-

"""
Module for recursive methods
"""

from numpy import zeros, identity, matmul, empty, insert, concatenate, power
from .models import polymodel
from .solvers import qrsolm

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
    #Caso o erro tenha variado menos que th entre as iterações, retorna False e sai do while
    if error[0,0] != 0 :
        for i in range(error.shape[1]):
            if (abs((error[0,i] - error[1,i])/error[0,i])) > th :
                return True
        return False
    else:
        return True
    
def emq(na,nb,nc,nk,u,y,th,n_max):
    """
    
    Performs the extended least squres algorithm on u,y data,
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
    # psi = zeros([y.shape[0]-w,nu*(nb+1)+ny*na])
    
    y_sol = y[w:] #saida usada para theta resoluçaõ do problema linear
    u_reg = zeros([u.shape[0]-w,nu*(nb+1)])
    cont = 0
    # nb quantos atrasos vou olhar
    # nk onde eu começo a olhar(-1, -2, -3,...)
    for c in range(nu): #para cada entrada
        for i in range((nb+1)): #u no regressor vai de u[k-nk] até u[k-nb-nk+1]
            u_reg[:,cont] = u[w-nbk+i:u.shape[0]-nbk+i,c]
            cont = cont + 1
    cont = 0
    y_reg = zeros([y.shape[0]-w,ny*na])
    for c in range(ny):
        for i in range(na):
            y_reg[:,cont] = -y[w-na+i:y.shape[0]-na+i,c]
            cont = cont + 1 
    psi = concatenate((u_reg,y_reg),axis=1)
    
    #MQ padrão : theta = pseudo_inversa(psi) * y
    theta_mq = qrsolm(psi,y_sol)
    
    # montar psi_emq (1° vez)
    
    psi_emq = zeros((y_sol.shape[0]-nc,psi.shape[1]+nc,ny))
    res = y_sol - matmul(psi,theta_mq)
    #Agora cada saida tem seu próprio regressor, já que os residuos são diferentes
    for i in range(ny):
        psi_emq[:,:psi.shape[1],i] = psi[nc:,:]
        for j in range(nc):
            res_i = res[j:-nc+j,i]
            psi_emq[:,psi_emq.shape[1]-nc+j,i] = res_i
    
    
    #Primeiro theta_emq
    theta_emq = zeros((psi_emq.shape[1],ny))
    
    #é preciso pegar o theta certo, o que foi calculado com o residuo da saida correspondente
    for i in range(ny):
        theta_emq[:,i] = qrsolm(psi_emq[:,:,i],y_sol[nc:,:])[:,i]
    
    #Iterações
    res = zeros((psi_emq.shape[0],nc))
    res_i = zeros((res.shape[0]))
    s_error = zeros((2,ny)) #uma linha pro erro passado e uma pro atual
    n_iteracoes = n_max
    i = 0
    while (var_erro(s_error) and i<n_iteracoes):
        for j in range(ny): #para cada psi
            res = (y_sol[nc:] - matmul(psi_emq[:,:,j],theta_emq[:,:])) #cada j tem o residuo para ir em um regressor
            for n in range(nc): #adição do residuo em psi_emq
                res_i[:-nc+n] = res[nc-n:,j]
                psi_emq[:,-nc+n,j] = res_i
            theta_emq[:,j] = qrsolm(psi_emq[:,:,j],y_sol[nc:,:])[:,j]
        
        s_error[1,:] = s_error[0,:]
        for ii in range(ny):
            s_error[0,ii] = sum(power(res,2)[:,ii])
            
        i = i + 1
        
    #Montagem das matrizes de polinomios 
    A = empty((ny,ny), dtype='object')
    B = empty((ny,nu), dtype='object')
    C = empty((ny,1),  dtype='object')
    
    nbp = nb+1
    
    #Preencher o cada elemento de B com zeros referentes a nk
    # Se nk é 3 B deve ser [0 0 bo b1 b2]
    
    for c in range(ny):
        for i in range(nu):
            if i == 0:
                B[c,i] = theta_emq[i*(nbp)-1+nbp::-1,c]
            else:
                B[c,i] = theta_emq[i*(nbp-1)+nbp:(nbp-1)*i:-1,c]
            B[c,i] = insert(B[c,i],0,zeros((nk))) #coloca zeros no começo
            
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
    
    #Crianção do objeto 
    n_params = theta_emq.shape[0]*theta_emq.shape[1]
    m = polymodel('armax',A,B,C,None,None,nk,n_params+1,(u,y),nu,ny,1)

    return m

def mqr(na,nb,nk,u,y):
    """

    Parameters
    ----------
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
    None. (will return the polymodel)

    """
    ny = y.shape[1]
    nu = u.shape[1]
    
    n_params = na+(nb+1)
    w = max(na,nb+nk)
    nb = nb+1 
    
    theta = zeros([n_params,ny]) #ja adequado ao mimo
    P = identity(n_params)*1e5 #caso mimo tera uma pk pra cada saida 
    
    #primeiros dados coloco manulmente para evitar um if no laço 
    u=u[w:,:]
    y=y[w:,:]
    
    cont = w
    psik = zeros((1,n_params))
    for i in range(y.shape[0]-w):
        for j in range(nu): 
            psik[0,:nb-1] = psik[0,1:nb]
        psik[0,nb-1] = u[cont-nk]
        for j in range(ny):
            psik[0,nb:nb+na-1] =  psik[0,nb+1:] #psi[cont-1,nb+1:]
        psik[0,nb+na-1] = -y[cont-1]
        K = matmul(P,psik.transpose())/((matmul( psik,matmul(P,psik.transpose()) ) + 1)[0,0])
        theta = theta + K*(y[cont]-matmul(psik,theta))
        #print(theta)
        P = P - K*(matmul(psik,P))
        # print(psik,u[cont],y[cont])
        cont = cont + 1
        
    
    # print(theta)
    
    #Montagem das matrizes de polinomios 
    A = empty((ny,ny), dtype='object')
    B = empty((ny,nu), dtype='object')
    #C = empty((ny,1),  dtype='object')
    
    nbp = nb+1
    
    #Preencher o cada elemento de B com zeros referentes a nk
    # Se nk é 3 B deve ser [0 0 bo b1 b2]
    
    for c in range(ny):
        for i in range(nu):
            if i == 0:
                B[c,i] = theta[nbp-2::-1,c]
            else:
                B[c,i] = theta[i*(nbp-1)+nbp:(nbp-1)*i:-1,c]
            B[c,i] = insert(B[c,i],0,zeros((nk))) #coloca zeros no começo
            
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
    
    #Crianção do objeto 
    n_params = theta.shape[0]*theta.shape[1]
    m = polymodel('arx',A,B,None,None,None,nk,n_params+1,(u,y),nu,ny,1)
    
    return m