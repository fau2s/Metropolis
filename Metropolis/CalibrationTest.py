from Metropolis import data
import numpy as np
from matplotlib import pyplot as pl
import os
from collections import defaultdict
from Metropolis import LSTMPrediction


def runtest(numoftests):
    for size in [21]:
        for Tid in [200]:
            for Tperf in [100]:
                testerrors = np.zeros((numoftests, 2))
                for test in range(numoftests):
                    lstmpred = LSTMPrediction()
                    testwithshift = data.Data(n=size, L=8, Tid=Tid, Tperf=Tperf, simulated=False, useShift=True, n_diff=1, model='log')
                    testwithshift.parameterEstimation()
                    testerrors[test, 1] = testwithshift.meanEstimationError()
                    noiselessWithShift = testwithshift.generateNoiselessFromA(30)
                    testwithshift.reconstructOriginal()
                    noiselessWithShiftRecon = testwithshift.recPperf
                    testwithshift.useShift = False
                    testwithshift.parameterEstimation()
                    testerrors[test, 0] = testwithshift.meanEstimationError()
                    noiselessWithOutShift = testwithshift.generateNoiselessFromA(30)
                    testwithshift.reconstructOriginal()
                    noiselessWithOutShiftRecon = testwithshift.recPperf
                    lstmpred.prediction(testwithshift.Pid)
                    tickList = '_'.join(testwithshift.ticker)
                    realdataPath = '.\\Test_' + tickList[:60] + '\\'
                    if not os.path.exists(realdataPath):
                        os.makedirs(realdataPath)
                    for t in range(size):
                        pl.clf()
                        pl.xlabel('Time [Days]')
                        pl.ylabel('Asset value [$]')
                        pl.title('Value of asset {asset} vs. estimated noiseless value'.format(asset=testwithshift.ticker[t][:-4]))
                        # pl.plot(noiselessWithShift[t, :30].transpose())
                        # pl.plot(noiselessWithOutShift[t, :30].transpose())
                        # pl.plot(testwithshift.Pperf[t, :30].transpose())
                        pl.plot(noiselessWithShiftRecon[t, :30].transpose())
                        pl.plot(noiselessWithOutShiftRecon[t, :30].transpose())
                        pl.plot(testwithshift.tr_Pperf[t, :30].transpose())
                        pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift', 'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])
                        pl.savefig(realdataPath + 'Real_estimated_N{i}_{s}.png'.format(i=size, s=testwithshift.ticker[t][:-4]))
                np.savetxt("calibrationTest_{0}_{1}_{2}_{3}_{4}.csv".format("Real", testwithshift.n, testwithshift.L, testwithshift.Tid, testwithshift.Tperf), testerrors, delimiter=',', fmt='%f')
                pl.clf()
                pl.plot(testerrors[:,0], testerrors[:,1], 'x', [np.min(testerrors), np.max(testerrors)], [np.min(testerrors), np.max(testerrors)], 'r--')
                pl.xscale("log")
                pl.yscale("log")
                # pl.plot(testerrors[:150, 0], testerrors[15:, 1], 'x')
                pl.xlabel('Error w/o shift')
                pl.ylabel('Error w/ shift')
                pl.title('Real data, population={0}, calibration={1}, trading={2}'.format(size, Tid, Tperf))
                # pl.show()
                pl.savefig("calibrationTest_{0}_{1}_{2}_{3}_{4}.png".format("Real", testwithshift.n, testwithshift.L, testwithshift.Tid, testwithshift.Tperf) )


def compareMethods(numoftests):
    methods = ['regular', 'diff', 'log', 'logdiff']
    Tperf = 100
    size = 493
    noiselessWithShiftRecon = defaultdict(np.array)
    for Tstart in range(900, 1000, 100):
        timerange = range(40, Tstart, 20)
        testerrors = np.zeros((numoftests * len(methods), 2 * len(timerange)))
        for ix, Tid in enumerate(timerange):
            for test in range(numoftests):
                testwithshift = data.Data(n=size, L=8, Tid=Tid, Tperf=Tperf, simulated=False, Tstart=Tstart, useShift=True,
                                          n_diff=1, model='regular')
                # tickList = '_'.join(testwithshift.ticker)
                # realdataPath = '.\\Test_' + tickList[:60] + '\\'
                realdataPath = '.\\Test_{0}_{1}_{2}'.format(Tstart, Tid, size) + '\\'
                if not os.path.exists(realdataPath):
                    os.makedirs(realdataPath)
                for no, method in enumerate(methods):
                    testwithshift.useShift = True
                    testwithshift.model = method
                    testwithshift.resetData()
                    testwithshift.selectModel(1)
                    testwithshift.makeZeroMean()
                    testwithshift.parameterEstimation()
                    testerrors[test*len(methods)+no, 1 + ix * 2] = testwithshift.meanEstimationError()
                    np.savetxt(
                    "eigenvalues_{0}_{1}_{2}_{3}_{4}.csv".format(method, testwithshift.n, testwithshift.Tstart,
                                                                 testwithshift.Tid, "shift"),
                    np.sort(np.abs(np.linalg.eigvals(testwithshift.A))))
                    # noiselessWithShift = testwithshift.generateNoiselessFromA(30)
                    noiselessWithShiftRecon[method] = testwithshift.recPperf
                    testwithshift.useShift = False
                    testwithshift.parameterEstimation()
                    testerrors[test*len(methods)+no, 0 + ix * 2] = testwithshift.meanEstimationError()
                    np.savetxt("eigenvalues_{0}_{1}_{2}_{3}_{4}.csv".format(method, testwithshift.n, testwithshift.Tstart, testwithshift.Tid,"noShift"),
                               np.sort(np.abs(np.linalg.eigvals(testwithshift.A))))
                # noiselessWithOutShiftRecon = testwithshift.recPperf
            for t in range(size):
                pl.clf()
                pl.xlabel('Time [Days]')
                pl.ylabel('Asset value [$]')
                pl.title('Value of asset {asset} vs. estimated noiseless value'.format(asset=testwithshift.ticker[t][:-4]))
                pl.plot(noiselessWithShiftRecon['regular'][t, :30].transpose())
                pl.plot(noiselessWithShiftRecon['diff'][t, :30].transpose())
                pl.plot(noiselessWithShiftRecon['log'][t, :30].transpose())
                pl.plot(noiselessWithShiftRecon['logdiff'][t, :30].transpose())
                pl.plot(testwithshift.tr_Pperf[t, :30].transpose())
                pl.legend(['TS with estimated AR matrix w/ regular model',
                           'TS with estimated AR matrix w/ diff model',
                           'TS with estimated AR matrix w/ log model',
                           'TS with estimated AR matrix w/ log diff model',
                           'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])
                pl.savefig(realdataPath + 'Real_estimated_N{i}_{s}.png'.format(i=size, s=testwithshift.ticker[t][:-4]))
        testerrors[testerrors==np.inf] = testerrors[testerrors!=np.inf].max()
        np.savetxt("calibrationTest_{0}_{1}_{2}.csv".format("Real", testwithshift.n, testwithshift.Tstart), testerrors, delimiter=',', fmt='%i')
            # pl.clf()
            # pl.plot(testerrors[:, 0], testerrors[:, 1], 'x', [np.min(testerrors), np.max(testerrors)],
            #         [np.min(testerrors), np.max(testerrors)], 'r--')
            # pl.xscale("log")
            # pl.yscale("log")
            # # pl.plot(testerrors[:150, 0], testerrors[15:, 1], 'x')
            # pl.xlabel('Error w/o shift')
            # pl.ylabel('Error w/ shift')
            # pl.title('Real data, population={0}, calibration={1}, trading={2}'.format(size, Tid, Tperf))
            # # pl.show()
            # pl.savefig("calibrationTest_{0}_{1}_{2}_{3}_{4}.png".format("Real", testwithshift.n, testwithshift.L,
            #                                                             testwithshift.Tid, testwithshift.Tperf))




if __name__ == '__main__':
    # runtest(1)
    compareMethods(1)