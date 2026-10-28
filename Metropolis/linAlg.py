import numpy as np
import scipy as sp

def autocovariance():
    return

def calcGeneralMatrix(p, A, B, G):
    result = np.zeros(A.shape)
    for i in range(p):
        result += np.matmul(B, np.matmul(G, np.transpose(A)))
    return result

def calcATGA(A, G):
    return np.matmul(A, np.matmul(G, np.transpose(A)))

def calcATGB(A, G, B):
    return np.matmul(B, np.matmul(G, np.transpose(A)))

def max_eigvector(A, B):
    # Identify the eigenvector corresponding to the largest eigenvalue of
    # the generalized eigenvalue equation A*x=lamb*B*x
    # @author R. Sipos
    from scipy import linalg as lina
    [V, D] = lina.eig(A, B)
    # return [D[:,0], V[0]]
    lamb = max(V)
    x = D[:, np.argmax(V)]
    return [x, lamb]


def getPermutationOrder(a, b):
    if a.size != b.size:
        return -1
    ix = np.zeros(a.size)
    for i in range(a.size):
        x = np.where(a == b[i])
        ix[i] = x[0]
    return ix.astype(int)