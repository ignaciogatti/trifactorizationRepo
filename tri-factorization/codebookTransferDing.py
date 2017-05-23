import numpy as np
from numpy import linalg as LA
from numpy.random import rand
import codebookConstruction as cdc
import trifactorizacionOrtogonal as trifactorization
   

def generateMasc(X):
    M = np.zeros(X.shape)
    M = X / X
    M = np.nan_to_num(M)
    return M

def transformMAtrix(F, G):
    n, m = F.shape
    Faux = np.zeros((n, m))
    for i in range(n):
        j = np.argmax(F[i])
        Faux[i][j] = 1.0
    
    #calculo Gaux
    n,m = G.shape
    Gaux = np.zeros((n,m))
    for i in range(n):
        j = np.argmax(G[i])
        Gaux[i][j] = 1.0
    return Faux, Gaux

def codebookTransfer(Xtgt, W, B, max_iter = 10):
    m, n = Xtgt.shape
    k, l = B.shape
    # defino Ftarget y Gtarget

    F = rand(m, k) 
    G = rand(n, l)        

    #calcular Ftarget y Gtarget
    for t in range(max_iter):
        
        G = trifactorization.calcularG( F, G, B, Xtgt)
        F = trifactorization.calcularF( F, G, B, Xtgt)
        
        Ftgt, Gtgt = transformMAtrix(F, G)

        Xaprox = np.dot(Ftgt, B)
        Gt = np.transpose(Gtgt)
        Xaprox = np.dot(Xaprox, Gt)
        Winv = 1 - W
        Xaprox = Winv * Xaprox
        XW = W * Xtgt
        Xaprox = XW + Xaprox
        ConvergeMatrix = Xtgt - Xaprox
        convergence = LA.norm(ConvergeMatrix)
        print("Convergencia...{}".format(convergence))

    return Ftgt, Gtgt

matrixAux = np.array( [ [2,3,3,3,2,3],[3,1,2,2,3,1],[3,1,2,2,3,1],[3,2,1,1,3,2],[2,3,3,3,2,3],[3,2,1,1,3,2] ] )
matrixTarget = np.array( [ [0,1,3,3,1],[3,3,2,0,3],[2,2,0,3,0],[1,1,3,0,0],[1,0,0,3,1],[3,0,2,2,3],[0,2,3,3,2] ] )
W = generateMasc( matrixTarget )
F, S, G = cdc.codebookConstruction( matrixAux, 3, 3 )

print("F...")
print(F)
print("G...")
print(G)
print("S...")

print(S)
Ftgt, Gtgt = codebookTransfer(matrixTarget, W, S)

print("Ftgt...")
print(Ftgt)
print("Gtgt...")
print(Gtgt)