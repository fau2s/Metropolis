import dcor
import numpy as np
from Metropolis import data


SP500 = data.Data(n=471, simulated=False, model='regular')


corrMatrix = np.zeros((SP500.P_all.shape[0], SP500.P_all.shape[0]))

for i in range(SP500.P_all.shape[0]):
    for j in range(SP500.P_all.shape[0]):
        print(i, j)
        corrMatrix[i, j] = dcor.distance_correlation(SP500.P_all[i, :], SP500.P_all[j, :])


aa = np.array([3.,-3., -4., 9., 1., 99., -55.]).reshape(7, 1)
bb = np.array([4., 4., 3., -6., 4., 4., 4.]).reshape(7, 1)
a = np.array([[1., 2., 3., 4.],
             [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]])
b = np.array([[1.], [0.], [0.], [1.]])
corr = dcor.distance_correlation(aa, bb)
print(corr)