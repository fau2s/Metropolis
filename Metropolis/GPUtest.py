from Metropolis import linAlg as la
import numpy as np
# import numba as nb
from numba import vectorize
from numba import cuda
from pyculib import rand as curand

@vectorize(['float32(float32, float32)'], target='cuda')
def Add(a, b):
    return a + b

N = 100000000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)

C = Add(A, B)

prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
rand = np.empty(10000)
prng.uniform(rand)
print(rand[:10])
# matrixSize = 10
#
# A = np.random.rand(matrixSize, matrixSize)
# G = np.random.rand(matrixSize, matrixSize)
# L = 5
#

def exhaustive(A, G, L):
    n = A.shape
    from itertools import combinations
    C = list(combinations(range(n[0]), L))
    numeig = len(C)
    maxeigval = 0
    maxeigvec = np.ones(L)
    eigpos = 1
    currval = np.zeros(numeig)
    # ATGA=A'*G*A;
    ATGA = calcATGA(A, G)  #### FIXED NF 20120802

    currval, Vcurr = f(ATGA, G, n, 0)

    if currval > maxeigval:
        maxeigval = currval
        maxeigvec = Vcurr
        # eigpos = j

    # reconstruct full w vector
    x = np.zeros(n[0])
    np.put(x, C[eigpos], maxeigvec)
    return x, maxeigval, currval, C

@vectorize(['float32(float32, float32)'], target='cuda')
def f(ATGA, G, n, i):
    diffVector = np.setdiff1d(range(n), i)
    ATGAcurr = np.delete(ATGA, diffVector, 0)
    ATGAcurr = np.delete(ATGAcurr, diffVector, 1)
    Gcurr = np.delete(G, diffVector, 0)
    Gcurr = np.delete(Gcurr, diffVector, 1)
    [Vcurr, currval] = max_eigvector(ATGAcurr, Gcurr)
    # currval = Dcurr

    return currval, Vcurr

@cuda.jit(device=True)
def calcATGA(A, G):
    return np.matmul(A, np.matmul(G, np.transpose(A)))

@cuda.jit(device=True)
def max_eigvector(A, B):
    from scipy import linalg as lina
    [V, D] = lina.eig(A, B)
    return [D[:, np.argmax(V)], max(V)]



