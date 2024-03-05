#This modules provides instrumental variables methods for discrete-time linear
#models
"""
Module for Instrumental Variables methods

@author: edumapurunga
"""

#%% imports
from numpy import dot, empty, sum, size, amax, concatenate, shape, zeros, insert
from scipy.linalg import solve
from numpy.linalg import matrix_rank

from pysid.identification.models import polymodel
from pysid.identification.solvers import qrsol
from pysid.io.check import chckin
from scipy.linalg import inv
from pysid.identification.pemethod import filtmat, arx
import warnings

#%% functions
__all__ = ['iv']

def sim_model(na,nb,nk,u,ny,A,B):
    N, nu = u.shape
    y = zeros((N,ny),dtype=float) #y's, com ny linhas e N colunas, cada linha é uma saida
    L = max(amax(na),amax(nb+nk)) #to know were to start
    for i in range(L,N):
        for j in range(ny): # for each output
            for k in range(ny): # to travel in cols of the Ao matrix
                # [::-1] makes the array backwards
                y[i,j] += dot(A[j,k][1:],-y[i-(len(A[j,k])-1):i,k][::-1])
            for k in range(nu):# for each input
                y[i,j] += dot(B[j,k][nk[j,k]:],u[i-len(B[j,k][nk[j,k]:]):i,k][::-1])
    return y


def iv2(na,nb,nk,u,y,method="u",new_experiment=None,prediction_method="arx"):
    """
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
    method : string, optional
        DESCRIPTION. The default is "u".
        string with the method that represents the instrument that should be used
        "u" : Only the input(s)
        "uy": The inputs and a prediction of y(t) based on a ARX first guess
        "ne": data from a new experiment
        
    new_experiment : touple, optional
        A touple contaning the two numpy arrays representing u and y, in this order
        of a new experiment
    prediction_method : string, optional
        DESCRIPTION. The default is "arx".
        string with the method that should be used to generate the prediction of 
        y when the method is 'uy'
    Returns
    -------
    None.

    """
    #theta_iv = (Z.T * phi)^-1 *Z.T * y
    # is_tuple = False #if its a tuple, then Z = [y_hat ... u]
    # if type(v) is tuple:
    #     is_tuple = True

    method_list = ["u","uy","ne"]
    prediction_method_list = ["arx","iv"]
    if method not in method_list:
        raise Exception("Not a valid method")
    if prediction_method not in prediction_method_list:
        raise Exception("Not a valid prediction method")
    if method == "ne" and new_experiment == None:
        raise Exception("Data of the new experiement must be passed")
    if new_experiment != None and method != "ne":
         warnings.warn("To use data from a new experiment use 'ne' as method\n")

    na, nb, _, _, _, nk, u, y = chckin(na, nb, [], [], [], nk, u, y)
    # Input handling
    Ny, ny = shape(y)
    Nu, nu = shape(u)
    nbk = nb + nk
    L = amax([amax(na), amax(nbk)])
    da = sum(sum(na))
    db = sum(sum(nb+1))
    nbp = nb+1
    # Np = sum(sum(na+nb+1))
    #d = da + db
    u_cont = 0
    y_cont = 0
    # Makes PHI
    phi = zeros((ny,), dtype = object)
    for i in range(ny):
        u_reg = zeros([Nu-L,int(sum(nb[i,:]+1))])
        y_reg = zeros([Ny-L,int(sum(na[i,:]))])
        for j in range(nu):
            for inb in range(nb[i,j]+1):
                u_reg[:,u_cont] = u[L-nbk[i,j]+inb:Nu-nbk[i,j]+inb,j]
                u_cont = u_cont + 1
        for jj in range(ny):
            for ina in range(na[i,jj]):
                y_reg[:,y_cont] = -y[L-na[i,jj]+ina:Ny-na[i,jj]+ina,jj]
                y_cont = y_cont + 1
        u_cont = 0
        y_cont = 0
        phi[i] = concatenate((u_reg,y_reg),axis=1)

    y_p = []
    u_collouns = []
    # prepare to construct Z based on the method
    if method == "uy":
        # use only the inputs do calculate a first model, then predicts a y_hat
        # and use that in Z
        if prediction_method == "iv":
            A,B = iv2(na, nb, nk, u, y, method="u")
        elif prediction_method == "arx":
            m = arx(na,nb,nk,u,y)
        y_p = sim_model(na, nb, nk, u, ny, m.A, m.B)
        # y_p = lfilter(B[0], A[0], u, axis=0)
        u_collouns  = db+1 #sum(nb+1) #ir até nb
        Lvi = amax([amax(nb+1),amax(na)])

    elif method == "u":
        for i in range(nu):
            u_collouns.append(sum(nb[:,i]+1)+sum(na[:,i]))
        Lvi = amax(u_collouns) #if Lvi > L: cut some rows from phi

    elif method == "ne":
        u_ne = new_experiment[0]
        y_ne = new_experiment[1]
        Ny_ne = y_ne.shape[0]
        Nu_ne = u_ne.shape[0]

        if ny != y_ne.shape[1]:
            raise Exception("The new experiment must have the same number of outputs as the original")
        if nu != u_ne.shape[1]:
            raise Exception("The new experiment must have the same number of inputs as the original")
        if Ny_ne != Nu_ne:
            raise Exception('Input and Output must be the same number of data samples')

        if Ny>Ny_ne:
            #corta do phi
            for i in range(ny):
                phi[i] = phi[i][(Ny-Ny_ne):,:]
        elif Ny_ne > Ny:
            for i in range(ny):
                phi[i] = phi[i][(Ny_ne-Ny):,:]

        Lvi = amax([amax(nb+1),amax(na)])

    if Lvi>L:
        for i in range(ny):
            phi[i] = phi[i][(Lvi-L):,:]
        w = Lvi
    else:
        w = L

    # Builds Z based on the method
    Z = zeros((ny,ny),dtype=list)
    nabp = na+nbp
    if method == "u":
        Zu  = zeros((ny,),dtype=list)
        n_lines = Nu - Lvi
        for i in range(ny):
            Zu  = zeros((n_lines,sum(nabp[i,:])),dtype=list)
            for j in range(ny):
                if i != j:
                    pass
                else:
                    for ii in range(nu):
                        for k in range(nabp[i,ii]):
                            if ii==0:
                                Zu[:,k] = u[k:Nu-Lvi+k,ii]
                            else:
                                Zu[:,sum(nabp[i,:ii])+k] = u[k:Nu-Lvi+k,ii]
                    Z[i,j] = Zu.astype(float)
        Z.astype(dtype=list)

    elif method == "uy":
        for i in range(ny):
            u_reg = zeros([Nu-w,int(sum(nb[i,:]+1))])
            y_reg = zeros([Ny-w,int(sum(na[i,:]))])
            for j in range(nu):
                for inb in range(nb[i,j]+1):
                    u_reg[:,u_cont] = u[w-nbk[i,j]+inb:Nu-nbk[i,j]+inb,j]
                    u_cont = u_cont + 1
            for jj in range(ny):
                for ina in range(na[i,jj]):
                    y_reg[:,y_cont] = -y_p[w-na[i,jj]+ina:Ny-na[i,jj]+ina,jj]
                    y_cont = y_cont + 1
            u_cont = 0
            y_cont = 0
            Z[i,i] = concatenate((u_reg,y_reg),axis=1)

    elif method == "ne":
        u_cont = 0
        y_cont = 0
        for i in range(ny):
            u_reg = zeros([Nu-w,int(sum(nb[i,:]+1))])
            y_reg = zeros([Ny-w,int(sum(na[i,:]))])
            for j in range(nu):
                for inb in range(nb[i,j]+1):
                    u_reg[:,u_cont] = u_ne[w-nbk[i,j]+inb:Nu-nbk[i,j]+inb,j]
                    u_cont = u_cont + 1
            for jj in range(ny):
                for ina in range(na[i,jj]):
                    y_reg[:,y_cont] = -y_ne[w-na[i,jj]+ina:Ny-na[i,jj]+ina,jj]
                    y_cont = y_cont + 1
            u_cont = 0
            y_cont = 0
            Z[i,i] = concatenate((u_reg,y_reg),axis=1)

    # Calculates theta
    theta_iv = zeros((ny,),dtype=object)
    for i in range(ny):
        theta_iv[i] = (inv(Z[i,i].T @ phi[i]) @ Z[i,i].T @ y[w:])[:,i]
        theta_iv[i] = theta_iv[i].reshape((len(theta_iv[i]),1))

    #Start to build the polymodel
    A = empty((ny,ny), dtype='object')
    B = empty((ny,nu), dtype='object')

    #Fill each element of B with zeros referring to nk
    # If nk = 2, B must be [0 0 bo b1 b2]
    for i in range(ny):
        for j in range(nu):
            if j == 0:
                B[i,j] = theta_iv[i][0:nbp[i,j]][::-1]
            else:
                B[i,j] = theta_iv[i][sum(nbp[i,0:j]):sum(nbp[i,0:j])+nbp[i,j]][::-1]
            B[i,j] = insert(B[i,j],0,zeros((nk[i,j]))) #fill with zeros

    for i in range(ny):
        for j in range(ny):
            if j==0:
                A[i,j] = theta_iv[i][sum(nbp[i,:]):sum(nbp[i,:])+na[i,j]][::-1]
            else:
                A[i,j] = theta_iv[i][sum(nbp[i,:])+sum(na[i,0:j]):sum(nbp[i,:])+sum(na[i,0:j])+na[i,j]][::-1]
            if i==j:
                A[i,j] = insert(A[i,j],0,1)
            else:
                A[i,j] = insert(A[i,j],0,0)

    # TODO : P...
    eh = (filtmat(A, y) - filtmat(B, u))[Lvi:Ny, 0:ny]
    sig = (eh.T @ eh)/Ny
    isig = inv(sig)
    Np = sum(sum(nabp))
    count_nc = 0
    count_nl = 0
    Rt = zeros((Np,Np))
    for i in range(ny):
        for j in range(ny):
            nl = Z[i,i].shape[1]
            nc = phi[j].shape[1]
            Rt[count_nl:count_nl+nl,count_nc:count_nc+nc] = Z[i,i].T @ phi[j]
            count_nc += nc
        count_nc = 0
        count_nl += nl
    R = Rt/Ny
    # R =   | Z1*phi1  Z1*phi2 | 
    # #     | Z2*phi1  Z2*phi2 |
    zlines = Z[0,0].shape[0]
    zcols = 0
    for i in range(Z.shape[0]):
        zcols += Z[i,i].shape[1]
    Z2 = zeros((ny*zlines,zcols))
    count_nc = 0
    count_nl = 0
    for i in range(ny):
        for j in range(ny):
            if i==j:
                nc = Z[i,i].shape[1]
                Z2[count_nl:count_nl+zlines, count_nc:count_nc+nc] = Z[i,i]
                count_nc += nc
                count_nl += zlines
    M = zeros((Np,Np),dtype=object)
    for k in range(0, Z2.shape[0], ny):
        M += Z2[k:k+ny].T @ isig @ Z2[k:k+ny]
    M /= Ny

    P = inv(R.T @ R) @ R.T @ M @ R @ inv(R.T @ R)
    m = polymodel('vi', A, B, None, None, None, nk, da+db, (u, y), nu, ny, 1)
    # TODO
    # m.M = M
    # m.setcov(...)
    # m.setparameters(...)
    return m
