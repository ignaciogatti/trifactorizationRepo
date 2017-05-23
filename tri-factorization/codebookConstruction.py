import numpy as np
import trifactorizacionOrtogonal as trifactorization


def codebookConstruction(X,irank,urank):

    F, S, G = trifactorization.onmftf(X, irank, urank, alpha=1.0, max_iter=200000, umbral_conv=0.00001)
    
    #calculo Faux
    n, m = F.shape
    Faux = np.zeros((n, m))
    for i in range(n):
        j = np.argmax(F[i])
        Faux[i][j] = 1.0
    print(Faux)
    
    #calculo Gaux
    print( G.shape)
    n,m = G.shape
    Gaux = np.zeros((n,m))
    for i in range(n):
        j = np.argmax(G[i])
        Gaux[i][j] = 1.0

    return Faux, S, Gaux


def defineCodebook(X, F, G):
    
    Ft = np.transpose(F)
    first_term = np.dot(Ft, X)
    first_term = np.dot( first_term, G)
    
    M_ones = np.ones( X.shape )
    second_term = np.dot(Ft, M_ones)
    second_term = np.dot(second_term, G)
    CB = first_term / second_term
    return CB

'''
matrix = np.array([[2,3,3,3,2,3],[3,1,2,2,3,1],[3,1,2,2,3,1],[3,2,1,1,3,2],[2,3,3,3,2,3],[3,2,1,1,3,2]])
F, S, G = codebookConstruction(matrix,3,3)    
print("Faux...")
print(F)
print("Gaux...")
print(G)
print("S...")
print(S)

CB = codebookConstructionCremonesi(matrix, F, G )
print("CB...")
print(CB)

'''