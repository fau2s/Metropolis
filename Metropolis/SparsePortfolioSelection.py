from Metropolis import PortfolioSelection as ps
from Metropolis import estimateMu as estMu
import numpy as np


def sparsePortfolioSelect(self, A, G, L, meth, C0, Pid, Tid):

    [xopt, lamb] = ps.sparse_meanrev_portfselect(A, G, L, meth)
    # runtime[rep,meth]=toc
    # lam[rep, meth - 1] = lamb
    if meth == 1:
        wexh = xopt  # save wexh for annealing.py comparison

    # Allow shorting, but display a warning message
    if np.sum(xopt >= 0) != A.shape[0]:
        print('Negative element in xopt - will need to do shorting')

    # Scale portfolio so that we sp   exactly C0
    xoptnorm = xopt / sum(abs(xopt))  # normalize xopt so that sum = 1
    xoptsp = C0 * xoptnorm  # amount of money to sp   on each asset
    xoptbuy = np.floor(xoptsp / Pid[:, Tid - 1])  # number of contracts to buy of each asset
    # Compute historical value of xoptbuy portfolio during Tid
    histportfval = np.matmul(xoptbuy, Pid)

    # Estimate the mu associated with xoptbuy
    mu0 = estMu.est_mu_linreg(histportfval)

    return xopt, lamb, xoptbuy, xoptnorm, mu0