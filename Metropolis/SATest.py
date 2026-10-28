from Metropolis import annealing as an
from Metropolis import data
import numpy as np
from timeit import default_timer as timer


def testSA():
    for n in range(10, 40, 5):
        for L in range(5, n-3):
            energyTimeTest = np.zeros((20, 6))
            for i in range(20):
                testdata = data.Data(n=n, L=L, Tid=350, Tperf=100, simulated=False, useShift=True)
                wexh = np.random.randint(low=1, high=100, size=n)
                wexh[:L] = 0
                saOld = an.anneal(wexh, wexh, A=testdata.A, G=testdata.G, L=L, T=10)
                # saOld.report = 1
                start = timer()
                [wOld, energyTimeTest[i, 0]] = saOld.process(True)
                # newparamOld = sa.neigh(wexh, sa.L, sa.T)
                # newenergyOld = sa.loss(sa.A, sa.G, newparamOld)
                energyTimeTest[i, 1] = (timer()-start)
                energyTimeTest[i, 2] = saOld.total
                saNew = an.anneal(wexh, wexh, A=testdata.A, G=testdata.G, L=L, T=10)
                # saNew.report = 1
                start = timer()
                [wNew, energyTimeTest[i, 3]] = saNew.process(False)
                # newparamNew = sa.neighNode(wexh, sa.L, sa.T)
                # newenergyNew = data.valami(newparamNew, sa.A, sa.G)
                energyTimeTest[i, 4] = (timer() - start)
                energyTimeTest[i, 5] = saNew.total
            print("Nincs ok a félelemre.")
            np.savetxt("SATest_{0}_{1}_{2}_{3}_{4}.csv".format("Real", testdata.n, testdata.L, testdata.Tid, testdata.Tperf), energyTimeTest, delimiter=',', fmt='%f')


if __name__ == '__main__':
    testSA()