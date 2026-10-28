# s(t+1)=A*s(t)+sigma*Z(t)
# Estimate sigma_squared given s and A. 
# @Author=N. Fogarasi

import numpy as np


def est_sigma2(s, A):
    size = s.shape
    
    summ = 0
    diff = np.zeros((size[1]-1, 1))
    for i in range(size[1]-1):
          d = s[:, i+1] - (A.dot(s[:, i]))
          diff[i] = d.dot(d)
          summ += diff[i]

    return summ / size[0] / size[1] #sigma2