# sparse_meanrev_portfselect.m
# Given matrices A, G and max cardinality L
# find all sparse subportfolios with max lamb
# using the method requested
# @Author: Norbert Forasi

import scipy as sp
import numpy as np
from Metropolis import annealing as ann
from Metropolis import linAlg as la


def sparse_meanrev_portfselect(A, G, L, method):
    n = A.shape
    if method == 1:  # Exhaustive
        [portf, lamb, num] = solve_exhaust(A, G, L)
        return [portf, lamb]
    else:
        if method == 4:  # Truncation
            [portf, lamb] = solve_trunc(A, G, L)
            return [portf, lamb]
        else:
             # Greedy search
            [portf, lamb] = solve_greedy(A, G, L)
            if method == 2:  # Greedy
                return [portf, lamb]

    # Simulated annealing.py
    # start from greedy solution
    initw = 1000*portf

    # Simulated annealing.py using new neigh function
    # start from greedy solution!!
    
    # opts = struct('Generator', @anneal_neigh2, 'MaxTries',3000,'MaxConsRej', 20000)
    an = ann.anneal(initw, 1000*portf, A, G, L)
    [wann3, dmax] = an.process()
    # [wann3, dmax] = ann.simulatedAnnealing(1, portf, 1000, A, G)
    # annealing.py minimizes, so store -dmax as the max.
    dsimann3 = -dmax
    if dsimann3 > lamb:
        lamb = dsimann3
        portf = wann3
             
    return [portf, lamb]


def max_eigvector(A, B):
    # Identify the eigenvector corresponding to the largest eigenvalue of
    # the generalized eigenvalue equation A*x=lamb*B*x
    # @author R. Sipos
    import scipy
    [V, D] = scipy.linalg.eig(A, B)
    lamb = max(V)
    x = D[V == lamb, :]
    return [x[0], lamb]


def solve_exhaust(A, G, L):
    # Exhaustive search to find solution of generalized eigenvalue problem
    # under cardinality constraint
    # @author: Norbert Fogarasi
    n = A.shape
    C = sp.misc.comb(range(n[0]), L)
    numeig = C.shape
    maxeigval = 0
    maxeigvec = np.ones(1, L)
    eigpos = 1
    currval = np.zeros(numeig[0], 1)
    # ATGA=A'*G*A;
    ATGA = la.calcATGA(A, G) #### FIXED NF 20120802
    
    for i in range(1,numeig[0]):
        ATGAcurr = ATGA[C[i, :], C[i, :]]
        Gcurr = G[C[i, :], C[i, :]]
        # [Vcurr,Dcurr]=eigs(ATGAcurr,Gcurr,1);
        [Vcurr, Dcurr] = max_eigvector(ATGAcurr, Gcurr)
        # ecord curr eigenvalue
        currval[i] = Dcurr
    if currval[i] > maxeigval:
         maxeigval = currval[i]
         maxeigvec = Vcurr
         eigpos = i
      
  
    # reconstruct full w vector
    x = np.zeros((n, 1))
    x[C[eigpos,:], 1] = maxeigvec
    lamb = (x*ATGA*x)/(x*G*x)
    return x, lamb


def solve_trunc(A, G, L):
    # Truncation method: take max L components of the eigenvector
    # and using those dimensions, find max eigenvalue
    # @Author: Norbert Fogarasi
    n = A.shape
    # ATGA = np.matmul(np.transpose(A), np.matmul(G, A))  # Fixed NF 20120802
    ATGA = la.calcATGA(A, G)
    [maxvec, dmax] = max_eigvector(ATGA, G)
    wsort = np.sort(maxvec)
    ix = getPermutationOrder(maxvec, wsort)
    trun_ix = ix[range(n[0] - L, n[0])]
    s_trun_ix = np.sort(trun_ix)
    # Find eigenval corresponding to truncated matrix
    if np.setdiff1d(range(n[0]), s_trun_ix).size == 0:
        return maxvec, dmax
    else:
        diffVector = np.setdiff1d(range(n[0]), s_trun_ix)
        Gcurr = np.delete(G, diffVector, 0)
        Gcurr = np.delete(Gcurr, diffVector, 1)
        #Gcurr = G[s_trun_ix, s_trun_ix]
        ATGAcurr = np.delete(ATGA, diffVector, 0)
        ATGAcurr = np.delete(ATGAcurr, diffVector, 1)
        #ATGAcurr = ATGA[s_trun_ix, s_trun_ix]
        [xopt, lamb] = max_eigvector(ATGAcurr, Gcurr)
        x = np.zeros(n[0])
        x[s_trun_ix] = xopt
        return x, lamb


def getPermutationOrder(a, b):
    if a.size != b.size:
        return -1
    ix = np.zeros(a.size)
    for i in range(a.size):
        x = np.where(a == b[i])
        ix[i] = x[0]
    return ix.astype(int)

def solve_greedy(A, G, sparsity):
    # Greedy algorithm to solve the generalized eigenvalue problem
    # under cardinatlity constraint
    dim = A.shape
    # fullATGA = np.matmul(np.transpose(A), np.matmul(G, A))
    # fullATGA = np.matmul(A, np.matmul(G, np.transpose(A))) # FIXED NF 20120802
    fullATGA = la.calcATGA(A, G)
    fullG = G
    if sparsity == dim[0]:
        [x,lamb] = max_eigvector(fullATGA, G)
        return x, lamb
    else:
        volmax = -np.inf
        xopt = np.zeros((sparsity, 1))
        I = np.array([], dtype=int)
        k = 0
        for k in range(sparsity):
            lastidx = -1
            for i in np.setdiff1d(range(dim[0]), I):
                Ii = np.insert(I, I.size, i)
                if Ii.size > 1:
                    diffVector = np.setdiff1d(range(dim[0]), Ii)
                    Gcurr = np.delete(fullG, diffVector, 0)
                    Gcurr = np.delete(Gcurr, diffVector, 1)
                    ATGAcurr = np.delete(fullATGA, diffVector, 0)
                    ATGAcurr = np.delete(ATGAcurr, diffVector, 1)
                    [x, vol] = max_eigvector(ATGAcurr, Gcurr)
                else:
                    ATGA = fullATGA[Ii, Ii]
                    G = fullG[Ii, Ii]
                    vol = ATGA / G
                    x = 1
                if vol > volmax:
                    volmax = vol
                    xopt = x
                    lastidx = i
            if lastidx == -1:
                break
            else:
                I = np.insert(I, I.size, lastidx) # ok<AGROW>
        # Revert to original space.
        fullx = np.zeros(dim[0])
        if sum(I.shape) != 0:
            fullx[I] = xopt
        x = fullx # disp('Vector x identified.');
        lamb = volmax
        return x, lamb