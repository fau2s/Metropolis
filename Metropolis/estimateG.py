import numpy as np
from Metropolis import estimateSigma2 as estS2
from Metropolis import linAlg as la
# s(t+1)=A*s(t)+sigma*Z(t)
# Estimate the correlation matrix G from above from a given time series s
# and given an estimate for A


def est_G_lyap_numeric_corrnoise(s, A):
    n = s.shape
    # first, estimate noise covariance
    sigma_sq_hat = est_noise_correl(s,A)

    # Solving the Lyapunov equation numerically
    # in order to ensure positive definiteness

    # paramters needed for running numerical sol
    delta = 0.1
    maxIter = 20000
    tol = 0.01
    lyapdist = np.zeros((maxIter, 1))
    # start with sample covariance
    G_new = np.cov(s)
    # ATGA = np.matmul(np.transpose(A), np.matmul(G_new, A))+sigma_sq_hat
    ATGA = la.calcATGA(A, G_new) + sigma_sq_hat
    bestlyap = np.linalg.norm(G_new - ATGA)
    G_best = G_new
    for i in range(maxIter):
        G_old = G_new
        G_new = G_old - delta * (G_old - ATGA)
        # check if G_new is posi defi. This will display error if not.
        # NOTE: chol appears unreliable for large eigenvalues, just means
        # G is divergent.
        try:
            R = np.linalg.cholesky(G_new)
        except:
            print('Divergent G - stopping numerical iteration.')
            break


        # check how we did...
        # ATGA = np.matmul(np.transpose(A), np.matmul(G_new, A))+sigma_sq_hat
        ATGA = la.calcATGA(A, G_new) + sigma_sq_hat
        lyapdist[i] = np.linalg.norm(G_new - ATGA)
        # save the best in case we become divergent...
        if (lyapdist[i] < bestlyap):
            G_best = G_new
            bestlyap = lyapdist[i]
        if np.linalg.norm(G_new - G_old) < tol:
            print('Converged to within movement tolerance.')
            break


    # figure;
    # plot (lyapdist);
    return G_best


# s(t+1)=A*s(t)+Z(t)
# Estimate Z(t) given s and A.
# @Author=N Fogarasi
def est_noise_correl(s,A):
    n = s.shape
    summ = np.zeros((n[0], n[0]))
    for i in range(n[1]-1):
          d = s[:, i+1] - A.dot(s[:, i])
          diff = np.outer(d, d)
          summ += diff
    return summ / n[1]


def est_G_lyap(s, A):
    # s(t+1)=A*s(t)+sigma*Z(t)
    # Estimate the correlation matrix G from above from a given time series s
    # and given an estimate for A
    dim = s.shape
    # first, estimate sigma
    sigma_sq_hat = estS2.est_sigma2(s, A) * np.eye(dim[0])
    # Solving the Lyapunov equation for G
    from scipy import linalg
    G_hat = linalg.solve_discrete_lyapunov(np.transpose(A), sigma_sq_hat)
    return G_hat