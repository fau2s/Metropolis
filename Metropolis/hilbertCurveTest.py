from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('testdata4NLP.txt')

p = 4
n = 2

hilbert_curve = HilbertCurve(p, n)
datamatrix = np.zeros((2**p, 2**p))

for start in range(len(data)):

    datarange = range(start, start+2**p*2**p)

    for coord, dist in zip(hilbert_curve.points_from_distances(range(2**p*2**p)), datarange):
        datamatrix[coord[0], coord[1]] = data[dist]

    plt.imshow(datamatrix)