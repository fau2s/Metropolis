import numpy as np
from matplotlib import pyplot as pl


def main():
    file = "E:\\PhD\\Metropolis\\var1TradingRange.csv"

    meansAndMinMaxs = np.loadtxt(file, delimiter='\t')

    minfit = np.polyfit(meansAndMinMaxs[:, 3], meansAndMinMaxs[:, 4], 3)
    maxfit = np.polyfit(meansAndMinMaxs[:, 5], meansAndMinMaxs[:, 6], 3)

    pl.plot(meansAndMinMaxs[:, 1], meansAndMinMaxs[:, 2], 'x')
    pl.plot([-500, 900],[-500, 900])
    pl.clf()
    pl.plot(meansAndMinMaxs[:, 3], meansAndMinMaxs[:, 4], 'x')
    pl.plot(meansAndMinMaxs[:, 3], np.polyval(minfit, meansAndMinMaxs[:, 3]), 'x')
    pl.plot([-500, 900], [-500, 900])
    pl.clf()
    pl.plot(meansAndMinMaxs[:, 5], meansAndMinMaxs[:, 6], 'x')
    pl.plot(meansAndMinMaxs[:, 5], np.polyval(maxfit, meansAndMinMaxs[:, 5]), 'x')
    pl.plot([-500, 900], [-500, 900])

    print(minfit, maxfit)

if __name__ == '__main__':
    main()