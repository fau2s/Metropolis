from Metropolis import estimateA as estA
from Metropolis import estimateG as estG
from Metropolis import estimateSigma2 as estS2
import numpy as np


def parameterEstimation(Pid, Pperf, A_true, sigma):
    # Pid = P_all[:, range(Tid)]
    # Pperf = P_all[:, range(Tid, Tid + Tperf)]

    #######################################
    # Step 1. Parameter Estimation
    #######################################
    A = estA.estA(Pid)
    sigma2 = estS2.est_sigma2(Pid, A)
    # display error just for debugging
    sigerr = sigma2 - sigma * sigma
    Aerr = np.linalg.norm(A_true - A)

    # estimate G
    G = estG.est_G_lyap(Pid, A)
    try:
        R = np.linalg.cholesky(G)
    except:
        # print('Exact Lyapunov solution is non posi defi. Using numerical solution.')
        G = estG.est_G_lyap_numeric_corrnoise(Pid, A)
    # G2 = np.cov(Pid)
    Beta = np.linalg.norm(G - np.cov(Pid))
    return A, G, Beta