import numpy as np
from numpy import linalg as LA
from numpy import random as rand
import codebookConstruction as cdc



def getMAE(Xtest, Xp):
    Xresult = np.abs(Xtest - Xp)
    mae = np.sum(Xresult)
    n_ratings = np.count_nonzero( Xtest )
    return ( mae/n_ratings )


def getRMSE(Xtest, Xp):
    Xresult = Xtest - Xp
    Xresult = Xresult**2
    rmse = np.sum(Xresult)
    n_ratings = np.count_nonzero( Xtest )
    rmse = rmse / n_ratings
    return ( np.sqrt( rmse ) ) 


def generateMask(X):
    M = np.zeros(X.shape)
    M = X / X
    M = np.nan_to_num(M)
    return M


def generateMaskTest(Xtest, Xp):
    Mtest = generateMask( Xtest )
    Mp = Xp * Mtest
    return Mp

def getFilledMatrix(Xtgt, W, Ftgt, B, Gtgt):
    Xaprox = np.dot(Ftgt, B)
    Gt = np.transpose(Gtgt)
    Xaprox = np.dot(Xaprox, Gt)
    Winv = 1 - W
    Xaprox = Winv * Xaprox
    XW = W * Xtgt
    Xaprox = XW + Xaprox
    return Xaprox



def codebookTransfer(Xtgt, W, B, max_iter = 10):
    n, m = Xtgt.shape
    k, l = B.shape
    # defino Ftarget y Gtarget
    Ftgt = np.zeros((n,k))
    Gtgt = np.zeros((m,l))
    
    #inicializo Ftarget
    for i in range(m):
        j = rand.randint(l)
        Gtgt[i][j] = 1.0
        
    #calcular Ftarget y Gtarget
    for t in range(max_iter):
        Gt = np.transpose(Gtgt)
        BG = np.dot(B,Gt)
        for i in range(n):
            Fvec = Xtgt[i] - BG
            Fnorm = LA.norm(Fvec, axis=1)
             
            j = np.argmin(Fnorm)
            
            Ftgt[i][:] = 0.0
            Ftgt[i][j] = 1.0
            
        FB = np.dot(Ftgt, B)
        Xt = np.transpose(Xtgt)
        FBt = np.transpose(FB)
        for i in range(m):
            Gvec = Xt[i] - FBt
            Gnorm = LA.norm( Gvec, axis=1)

#            print("theta...")
 #           print(theta.shape)
#            Gvec[j] = LA.norm(theta)
            j = np.argmin( Gnorm )
            Gtgt[i][:] = 0.0
            Gtgt[i][j] = 1.0
    
        Xaprox = getFilledMatrix(Xtgt, W, Ftgt, B, Gtgt)
        ConvergeMatrix = Xtgt - Xaprox
        convergence = LA.norm(ConvergeMatrix)
        print("Convergencia...{}".format(convergence))

    return Ftgt, Gtgt


'''
matrixAux = np.array( [ [2,3,3,3,2,3],[3,1,2,2,3,1],[3,1,2,2,3,1],[3,2,1,1,3,2],[2,3,3,3,2,3],[3,2,1,1,3,2] ] )
matrixTarget = np.array( [ [0,1,3,3,1],[3,3,2,0,3],[2,2,0,3,0],[1,1,3,0,0],[1,0,0,3,1],[3,0,2,2,3],[0,2,3,3,2] ] )
matrixTargetHide = np.array( [ [0,1,3,0,1],[3,3,0,0,3],[2,2,0,3,0],[0,1,3,0,0],[1,0,0,3,1],[3,0,2,0,3],[0,2,0,3,2] ] )

W = generateMask( matrixTargetHide )
F, S, G = cdc.codebookConstruction( matrixAux, 3, 3 )

print("F...")
print(F)
print("G...")
print(G)
print("S...")

CB = cdc.defineCodebook( matrixAux, F, G )

print(CB)
Ftgt, Gtgt = codebookTransfer( matrixTargetHide, W, CB )

print("Ftgt...")
print(Ftgt)
print("Gtgt...")
print(Gtgt)

Xaprox = getFilledMatrix( matrixTargetHide, W, Ftgt, CB, Gtgt )
print('Xtgt...')
print( Xaprox )

Xaprox = generateMaskTest( matrixTarget, Xaprox )
mae = getMAE( matrixTarget, Xaprox )

rmse = getRMSE( matrixTarget, Xaprox )

print('mae...{}'.format( mae ) )
print('rmse...{}'.format( rmse ) )

np.savetxt('/home/ignacio/Datasets/matrixTarget.txt', matrixTarget )
np.savetxt('/home/ignacio/Datasets/matrixAprox.txt', Xaprox )

#'''