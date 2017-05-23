import numpy as np
from numpy import nan_to_num
from numpy.random import rand
from numpy import linalg as LA


def calcularG( F, G, S, X):
    
    Xt = np.transpose(X)
    FS = np.dot(F, S)
    N = np.dot(Xt, FS)
    Gt = np.transpose(G)
    D = np.dot(G, Gt)
    D = np.dot(D, N)
    Gnew = N/D
    Gnew = np.sqrt(Gnew)
    Gnew = nan_to_num( G * Gnew)

    return Gnew
        

def calcularF( F, G, S, X):
    
    St = np.transpose(S)
    GSt = np.dot(G, St)
    N = np.dot(X, GSt)
    Ft = np.transpose(F)
    D = np.dot(F, Ft)
    D = np.dot(D, N)
    Fnew = N / D
    Fnew = np.sqrt(Fnew)
    Fnew = nan_to_num( F * Fnew)

    return Fnew
    
def calcularS( F, G, S, X):
    
    Ft = np.transpose(F)
    N = np.dot(Ft, X)
    N = np.dot(N,G)
    D = np.dot(Ft, F)
    D = np.dot(D, S)
    Gt = np.transpose(G)
    GGt = np.dot(Gt,G)
    D = np.dot(D, GGt)
    Snew = N / D
    Snew = np.sqrt(Snew)
    Snew = nan_to_num( S * Snew)
    return Snew

def onmftf(X, item_rank, user_rank, alpha=1.0, max_iter=1000, umbral_conv=0.5):
        """
        Fast non-negative matrix tri factorization.
        Parameters
        ----------
        X: array [m x n]
            Data matrix.
        user_rank: int
            Maximum rank of the factor model.
        item_rank: int
            Maximum rank of the factor model.
        alpha: int
            Orthogonality regularization parameter.
        max_iter: int
            Maximum number of iterations.
        Returns
        F: array [m x rank]
            Coefficient matrix (row clustering).
        S: array [rank x n]
            Basis matrix (column clustering / patterns).
        G: array [rank x n]
            Basis matrix (column clustering / patterns).
        """

        m, n = X.shape
        F = rand(m, user_rank) 
        G = rand(n, item_rank)        
        S = rand(user_rank, item_rank)

        
        convergence = 1000.0
        i = 0
        while (convergence > umbral_conv) and (i < max_iter):
            
            #actualizo las matrices
            G = calcularG( F, G, S, X)           
            F = calcularF( F, G, S, X)
            S = calcularS( F, G, S, X)
            
            #analizo convergencia
            Xaprox = np.dot(F,S)
            Gt = np.transpose(G)
            Xaprox = np.dot(Xaprox,Gt)
            ConvergeMatrix = X - Xaprox
            convergence = LA.norm(ConvergeMatrix)
            #print("Convergencia...{}".format(convergence))
            i+=1
        print("#iteraciones...{}".format(i))
        return F, S, G

matrix = np.array([[2,3,3,3,2,3],[3,1,2,2,3,1],[3,1,2,2,3,1],[3,2,1,1,3,2],[2,3,3,3,2,3],[3,2,1,1,3,2]])
F, S, G = onmftf(matrix,3,3)
print("F shape {}".format(F.shape))
print(F)
print("G shape {}".format(G.shape))
print(G)
print("S shape {}".format(S.shape))
print(S)