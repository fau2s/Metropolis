from Metropolis import estimateMu as estMu
from Metropolis import parameterEstimation as pe
from Metropolis import data
from Metropolis import Portfolio as pf
from Metropolis import SparsePortfolioSelection as sps
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from Metropolis import LSTMPrediction
from scipy.signal import argrelextrema
from Metropolis import LuongAttention

class Trading:
    def __init__(self, n, spars, simulated, idTimeWindow, tradingStart, tradingTimeWindow, useShift, n_diff, model, sigma=0.001):
        self.timeSeries = data.Data(n=n, L=spars, Tid=idTimeWindow, Tperf=tradingTimeWindow, Tstart=tradingStart, simulated=simulated, useShift=useShift, n_diff=n_diff, model=model, sigma=sigma)
        self.portfolios = pf.Portfolio(self.timeSeries)
        self.timeSeriesFileName = "timeseries_{0}_{1}_{2}_{3}_{4}_{5}.csv".format(n, spars, idTimeWindow, tradingTimeWindow, "Simulated" if simulated else "Real", "withShift" if useShift else "noShift")
        self.cashFileName = "cash_{0}_{1}_{2}_{3}_{4}_{5}.csv".format(n, spars, idTimeWindow, tradingTimeWindow, "Simulated" if simulated else "Real", "withShift" if useShift else "noShift")
        # self.meanErrorFileName = "meanError_{0}_{1}_{2}_{3}_{4}.csv".format(n, spars, idTimeWindow, "Simulated" if simulated else "Real", "withShift" if useShift else "noShift")
        # self.portfolios = [pf.Portfolio(self.timeSeries), pf.Portfolio(self.timeSeries)]
        # self.xoptbuy0 = xoptbuy
        self.portfheld = False
        self.selled = False
        self.risktaker = True
        self.buyt = np.array([], dtype=int)
        self.buyp = np.array([], dtype=float)
        self.sellt = np.array([], dtype=int)
        self.sellp = np.array([], dtype=float)
        self.mu = np.zeros(self.timeSeries.Tperf + 1)
        self.C = np.zeros(self.timeSeries.Tperf + 1)  # amount of cash held at time i
        self.x = np.zeros((self.timeSeries.Tperf + 1, self.timeSeries.n))  # number of assets held
        self.V = np.zeros(self.timeSeries.Tperf + 1)  # value of assets    held at time i
        self.numtrade = 0  # number of trades performed
        # self.mu0 = self.portfolios.mu0
        self.mu0 = np.mean(self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, -30:-1]))
        self.eps1 = self.mu0 - 0.7*np.std(self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, -30:-1]))
        self.eps2 = self.mu0 + 0.7*np.std(self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, -30:-1]))

        self.ret = 0
        self.profit = 0

        self.tradingTimeWindow = tradingTimeWindow


    def Idivergence(self, x, y):
        return np.sum(x * np.log2(x/y))


    def OUfromFokkerPlanck(self, alpha, D, t, s, x, y, mean):
        inflation = np.exp(-alpha*(t-s))
        inflationsq = 1 - inflation * inflation
        c = alpha / (2 * D * inflationsq)
        return np.sqrt(c/np.pi)*np.exp(-c*(x-(inflation*y + (1-inflation)*mean)) ** 2)

    def regularMeanRev(self):
        pass

    def onlineVAR(self, isBuy):
        if isBuy:
            calibData = self.portfolios.xoptbuy.dot(self.timeSeries.P_all[:, self.timeSeries.Tstart + self.time-30:self.timeSeries.Tstart + self.time-1])
            self.mu0 = np.mean(calibData)
            self.tradingRange = 0.7 * np.std(calibData)
            self.eps1 = self.mu0 - self.tradingRange
            self.eps2 = self.mu0 + self.tradingRange
        else:
            self.eps2 = self.eps1 + (self.eps2 - self.eps1) * 0.95

        currentPortfolioValue = self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf[:, self.localTime])
        predictedValue = self.portfolios.xoptbuy0.dot(self.timeSeries.generateFromOnline(2, self.localTime)[:-1, :])

        if isBuy:
            toTrade = (self.eps1 > currentPortfolioValue) & (predictedValue[-2] < predictedValue[-1])
        else:
            toTrade = ((self.eps2 < currentPortfolioValue) & (predictedValue[-2] > predictedValue[-1])) | ((currentPortfolioValue - self.buyp[-1] < 0.5 * self.tradingRange) & (self.time - (self.buyt[-1] - self.timeSeries.Tid) > 10))
        return toTrade

    def onlineLSTM(self, isBuy):
        # lstmpred = LSTMPrediction.LSTMPrediction(np.concatenate((self.portfolios.xoptbuy0.dot(self.timeSeries.Pid).reshape(
        #     self.timeSeries.Tid, 1), self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf).reshape(self.timeSeries.Tperf, 1))),self.timeSeries.Tid/(self.timeSeries.Tid+self.timeSeries.Tperf))

        if isBuy:
            calibData = self.portfolios.xoptbuy.dot(self.timeSeries.P_all[:, self.timeSeries.Tstart + self.time-30:self.timeSeries.Tstart + self.time-1])
            self.mu0 = np.mean(calibData)
            self.tradingRange = 0.7 * np.std(calibData)
            self.eps1 = self.mu0 - self.tradingRange
            self.eps2 = self.mu0 + self.tradingRange
        else:
            self.eps2 = self.eps1 + (self.eps2 - self.eps1) * 0.95

        currentPortfolioValue = self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf[:, self.localTime])
        # lstmtrainedPortfolio, predictedValue, lstmpredictedPortfolio2 = self.lstmpred.prediction(self.timeSeries.Tid / (self.timeSeries.Tid + self.timeSeries.Tperf))

        pred_len = len(self.predictedValue) - 1

        if isBuy:
            toTrade = (self.eps1 > currentPortfolioValue) & (self.predictedValue[min(max(self.localTime-2, 0), pred_len)] < self.predictedValue[min(max(self.localTime-1, 0), pred_len)])
        else:
            toTrade = ((self.eps2 < currentPortfolioValue) & (self.predictedValue[min(max(self.localTime-2, 0), pred_len)] > self.predictedValue[min(max(self.localTime-1, 0), pred_len)])) |\
                      ((currentPortfolioValue - self.buyp[-1] < 0.5 * self.tradingRange) & (self.time - (self.buyt[-1] - self.timeSeries.Tid) > 300) & (self.predictedValue[min(max(self.localTime-2, 0), pred_len)] > self.predictedValue[min(max(self.localTime-1, 0), pred_len)]))
        return toTrade

    def offlineVAR(self, isBuy):
        predictedData = self.portfolios.xoptbuy0.dot(self.timeSeries.generateNoiselessFromA(self.timeSeries.Tperf, self.timeSeries.Pperf[:, 0])[:-1,:])
        if isBuy:
            # calibData = self.portfolios.xoptbuy.dot(self.timeSeries.P_all[:, self.timeSeries.Tstart + self.time-30:self.timeSeries.Tstart + self.time-1])
            self.mu0 = np.mean(predictedData)
            self.tradingRange = 0.7 * np.std(predictedData)
            self.eps1 = self.mu0 - self.tradingRange
            self.eps2 = self.mu0 + self.tradingRange
        else:
            self.eps2 = self.eps1 + (self.eps2 - self.eps1) * 0.95

        currentPortfolioValue = self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf[:, self.localTime])
        predictedValue = self.portfolios.xoptbuy0.dot(self.timeSeries.generateFromOnline(2, self.localTime)[:-1, :])

        if isBuy:
            toTrade = (self.eps1 > currentPortfolioValue) & (predictedValue[-2] < predictedValue[-1])
        else:
            toTrade = ((self.eps2 < currentPortfolioValue) & (predictedValue[-2] > predictedValue[-1])) | ((currentPortfolioValue - self.buyp[-1] < 0.5 * self.tradingRange) & (self.time - (self.buyt[-1] - self.timeSeries.Tid) > 200))
        return toTrade

    def offlineLSTM(self, isBuy):
        if isBuy:
            self.mu0 = np.mean(self.offlinePredictedValue)
            self.tradingRange = 0.7 * np.std(self.offlinePredictedValue)
            self.eps1 = self.mu0 - self.tradingRange
            self.eps2 = self.mu0 + self.tradingRange
        else:
            self.eps2 = self.eps1 + (self.eps2 - self.eps1) * 0.95

        currentPortfolioValue = self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf[:, self.localTime])

        pred_len = len(self.offlinePredictedValue) - 1

        if isBuy:
            toTrade = (self.eps1 > currentPortfolioValue) & (self.offlinePredictedValue[min(max(self.localTime-2, 0), pred_len)] < self.offlinePredictedValue[min(max(self.localTime-1, 1), pred_len)])
        else:
            toTrade = ((self.eps2 < currentPortfolioValue) &
                       (self.offlinePredictedValue[min(max(self.localTime-2,0), pred_len)] > self.offlinePredictedValue[min(max(self.localTime-1, 1), pred_len)])) |\
                      ((currentPortfolioValue - self.buyp[-1] < 0.5 * self.tradingRange) &
                       (self.time - (self.buyt[-1] - self.timeSeries.Tid) > 300) &
                       (self.offlinePredictedValue[min(max(self.localTime-2, 0), pred_len)] > self.offlinePredictedValue[min(max(self.localTime-1, 1), pred_len)]))
        return toTrade

    def plotPortfolioAndTrade(self, strategy, wasTrade=False):
        lookback_days = 3
        pl.clf()
        pl.xlabel('Time [Days]')
        pl.ylabel('Portfolio value [$]')
        pl.title('Value of portfolio during trading period')
        pl.plot(range(self.timeSeries.Tid), self.portfolios.xoptbuy0.dot(self.timeSeries.Pid),'g-')
        pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid+self.timeSeries.Tperf), self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf),'r-')
        if strategy == "OnlineVar":
            pl.plot(range(self.timeSeries.Tid+1, self.timeSeries.Tid+1+self.timeSeries.Tperf), self.portfolios.xoptbuy0.dot(self.timeSeries.generateFromOnline(self.timeSeries.Tperf, 0)[:-1, :]),'c-')
        elif strategy == "OnlineLSTM":
            pl.plot(range(self.timeSeries.Tid + int(np.floor((1 + lookback_days) / 2)),
                      self.timeSeries.Tid + self.timeSeries.Tperf - int(np.ceil((1 + lookback_days) / 2))),
                self.predictedValue, 'y-')
        elif strategy == "OfflineVar":
            pl.plot(range(self.timeSeries.Tid + 1, self.timeSeries.Tid + 1 + self.timeSeries.Tperf),
                    self.portfolios.xoptbuy0.dot(self.timeSeries.generateNoiselessFromA(self.timeSeries.Tperf, self.timeSeries.Pperf[:, 0])[:-1, :]),
                    'c-')
        elif strategy == "OfflineLSTM":
            pl.plot(range(self.timeSeries.Tid + int(np.floor((1 + lookback_days) / 2)),
                      self.timeSeries.Tid + self.timeSeries.Tperf - int(np.ceil((1 + lookback_days) / 2))),
                self.offlinePredictedValue, 'y-')

        if wasTrade:
            pl.plot(self.buyt[-1], self.buyp[-1], 'xk')
            pl.plot(self.sellt[-1], self.sellp[-1], 'ok')
        # pl.show()

    def processOnOffline(self, strategy):
        # Check whether to trade immediatel

        self.time = 0

        buyingDictionary = {"regularMeanRev": lambda x: x+1,
                            "OnlineVar": self.onlineVAR,
                            "OfflineVar": self.offlineVAR,
                            "OnlineLSTM": self.onlineLSTM,
                            "OfflineLSTM": self.offlineLSTM
                            }

        sellingDictionary = {"regularMeanRev": lambda x: x+1,
                            "OnlineVar": self.onlineVAR,
                            "OfflineVar": self.offlineVAR,
                            "OnlineLSTM": self.onlineLSTM,
                            "OfflineLSTM": self.offlineLSTM
                             }


        lookback_days = 3
        self.timeSeries.parameterEstimation()
        if strategy in ["OnlineLSTM", "OfflineLSTM"]:
            self.lstmpred = LSTMPrediction.LSTMPrediction(np.concatenate((self.portfolios.xoptbuy0.dot(self.timeSeries.Pid).reshape(
                self.timeSeries.Tid, 1), self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf).reshape(self.timeSeries.Tperf, 1))),self.timeSeries.Tid/(self.timeSeries.Tid+self.timeSeries.Tperf))
            lstmtrainedPortfolio, self.predictedValue, self.offlinePredictedValue = self.lstmpred.prediction(
                self.timeSeries.Tid / (self.timeSeries.Tid + self.timeSeries.Tperf))

            # lstmtrainedPortfolio, lstmpredictedPortfolio, lstmpredictedPortfolio2 = lstmpred.prediction(self.timeSeries.Tid/(self.timeSeries.Tid+self.timeSeries.Tperf))
        # locmin = argrelextrema(lstmpredictedPortfolio, np.less)
        # locmax = argrelextrema(lstmpredictedPortfolio, np.greater)
        # self.mu0 = np.mean(lstmpredictedPortfolio)
        # self.eps1 = self.mu0 - 0.7 * np.std(lstmpredictedPortfolio)
        # self.eps2 = self.mu0 + 0.7 * np.std(lstmpredictedPortfolio)

        # self.plotPortfolioAndTrade(strategy, False)
        # pl.clf()
        # pl.xlabel('Time [Days]')
        # pl.ylabel('Portfolio value [$]')
        # pl.title('Value of portfolio during trading period')
        # pl.plot(range(self.timeSeries.Tid), self.portfolios.xoptbuy0.dot(self.timeSeries.Pid),'g-')
        # pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid+self.timeSeries.Tperf),   self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf),'r-')
        # pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid+self.timeSeries.Tperf), self.portfolios.xoptbuy0.dot(self.timeSeries.generateNoiselessFromA(self.timeSeries.Tperf, self.timeSeries.Pperf[:, 0])[:-1, :]),'b-')
        # pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid+self.timeSeries.Tperf), self.portfolios.xoptbuy0.dot(self.timeSeries.generateFromOnline(self.timeSeries.Tperf, 0)[:-1, :]),'c-')
        # pl.plot(range(self.timeSeries.Tid+int(np.floor((1+lookback_days)/2)), self.timeSeries.Tid+self.timeSeries.Tperf-int(np.ceil((1+lookback_days)/2))), lstmpredictedPortfolio,'y-')
        # pl.plot(self.timeSeries.Tid+int(np.floor((1+lookback_days)/2))+locmin[0], lstmpredictedPortfolio[locmin],'x')
        # pl.plot(self.timeSeries.Tid+int(np.floor((1+lookback_days)/2))+locmax[0], lstmpredictedPortfolio[locmax],'o')
        # pl.plot([0, self.timeSeries.Tid+self.timeSeries.Tperf],[self.eps1, self.eps1], '--')
        # pl.plot([0, self.timeSeries.Tid+self.timeSeries.Tperf],[self.eps2, self.eps2], '--')
        # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])

        # pl.show()

        # fig = pl.gcf()
        # fig.set_size_inches(18.5, 10.5, forward=True)
        # tickList = '_'.join(self.timeSeries.ticker)
        # realdataPath = '.\\Portfolio_N{i}_{s}'.format(i=self.timeSeries.n, s=self.timeSeries.L) + tickList[:60] + '.png'
        # pl.savefig(realdataPath)

        self.time = 0
        self.localTime = 0

        self.timeSeries.parameterEstimation()
        # if buyingDictionary[strategy](True):
        #     lstmpred = LSTMPrediction.LSTMPrediction(
        #         np.concatenate((self.portfolios.xoptbuy0.dot(self.timeSeries.Pid).reshape(
        #             self.timeSeries.Tid, 1),
        #                         self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf).reshape(self.timeSeries.Tperf, 1))),
        #         self.timeSeries.Tid / (self.timeSeries.Tid + self.timeSeries.Tperf))
        #     lstmtrainedPortfolio, lstmpredictedPortfolio, lstmpredictedPortfolio2 = lstmpred.prediction(
        #         self.timeSeries.Tid / (self.timeSeries.Tid + self.timeSeries.Tperf))
        #     locmin = argrelextrema(lstmpredictedPortfolio, np.less)
        #     locmax = argrelextrema(lstmpredictedPortfolio, np.greater)
        # self.mu0 = np.mean(lstmpredictedPortfolio)
        # self.eps1 = self.mu0 - 0.7 * np.std(lstmpredictedPortfolio)
        # self.eps2 = self.mu0 + 0.7 * np.std(lstmpredictedPortfolio)
        pd.DataFrame(self.timeSeries.tr_P_all).to_csv(self.timeSeriesFileName, mode='a')

        # pd.DataFrame(np.array([self.timeSeries.n, self.timeSeries.L, self.timeSeries.simulated, self.timeSeries.useShift, self.timeSeries.meanEstimationError()])).to_csv(self.meanErrorFileName)
        if buyingDictionary[strategy](True):
            V0 = self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, self.timeSeries.Tid - 1])
            # print('0. BUY at price ', str(V0))
            self.portfheld = True
            self.buyt = np.insert(self.buyt, self.buyt.size, self.timeSeries.Tid)
            self.buyp = np.insert(self.buyp, self.buyp.size, V0)
            self.C[0] = self.portfolios.C0 - V0
            self.x[0, :] = self.portfolios.xoptbuy
            self.numtrade += 1
        else:
            self.C[0] = self.portfolios.C0
            self.x[0, :] = np.zeros(self.timeSeries.n)
            # V0 = 0

        while self.time < self.tradingTimeWindow:
            if self.selled:
                # self.dump(self.mu[i-1], self.C[i-1])
                self.timeSeries.parameterEstimation()
                self.portfolios.selection()
                if strategy in ["OnlineLSTM", "OfflineLSTM"]:
                    if self.timeSeries.Tperf < lookback_days + 2:
                        break
                    self.lstmpred = LSTMPrediction.LSTMPrediction(
                        np.concatenate((self.portfolios.xoptbuy0.dot(self.timeSeries.Pid).reshape(
                            self.timeSeries.Tid, 1), self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf).reshape(
                            self.timeSeries.Tperf, 1))),
                        self.timeSeries.Tid / (self.timeSeries.Tid + self.timeSeries.Tperf))
                    lstmtrainedPortfolio, self.predictedValue, self.offlinePredictedValue = self.lstmpred.prediction(
                        self.timeSeries.Tid / (self.timeSeries.Tid + self.timeSeries.Tperf))
                self.selled = False
                # self.plotPortfolioAndTrade(strategy, False)
                self.localTime = 0
                # self.mu[self.time] = self.portfolios.mu0
                # self.eps1 = self.mu[self.time] - np.std(self.portfolios.xoptbuy0.dot(self.timeSeries.Pid[:, 20:]))
                # self.eps2 = self.mu[self.time] + np.std(self.portfolios.xoptbuy0.dot(self.timeSeries.Pid[:, 20:]))
            # for i in range(Tperf):
            # compute mu in terms of orig portfolio
            # mu[i] = estMu.est_mu_linreg(xoptbuy0.dot(P_all[:, range(Tid + i)]))
            self.portfolios.est_mu_linreg()
            # self.mu[i] = self.portfolios.mu0
            # self.mu[self.time] = np.mean(self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, -30:-1]))
            currval = self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf[:, self.localTime])
            # df = pd.DataFrame({ 'lambda': self.portfolios.weights['annealing'][1]}, index=[0])
            # df.to_csv("eigenval_13_10_160_100.csv", mode='a')
            if not self.portfheld:
                # check if we should buy

                if buyingDictionary[strategy](True):
                    self.portfheld = True
                    if self.risktaker:
                        # Case 1: wager all our money
                        self.portfolios.xoptsp = self.C[self.time] * self.portfolios.xoptnorm

                    # Case 2: we only spend C0, so do not recompute xoptsp
                    self.portfolios.xoptbuy = np.floor(self.portfolios.xoptsp / self.timeSeries.tr_Pperf[:, self.localTime])
                    self.V[self.time] = self.portfolios.xoptbuy.dot(self.timeSeries.Pperf[:, self.localTime])
                    self.C[self.time] = self.C[self.time] - self.V[self.time]
                    self.C[self.time + 1] = self.C[self.time]
                    self.x[self.time + 1, :] = self.portfolios.xoptbuy
                    self.numtrade += 1
                    # print([str(i), '. BUY at price ', str(V[i])])
                    self.buyt = np.insert(self.buyt, self.buyt.size, self.timeSeries.Tid + self.time)
                    self.buyp = np.insert(self.buyp, self.buyp.size, currval)
                else:
                    # keep money
                    self.C[self.time + 1] = self.C[self.time]
                    self.V[self.time] = 0
                    self.x[self.time + 1, :] = self.x[self.time, :]

            else:
                # compute portf value at new price
                self.V[self.time] = self.x[self.time, :].dot(self.timeSeries.Pperf[:, self.localTime])
                # check if we should sell
                if sellingDictionary[strategy](False):
                    self.portfheld = False
                    self.selled = True
                    self.C[self.time + 1] = self.C[self.time] + self.x[self.time, :].dot(self.timeSeries.Pperf[:, self.localTime])
                    self.V[self.time + 1] = 0
                    self.x[self.time + 1, :] = np.zeros(self.timeSeries.n)
                    self.numtrade += 1
                    # print([str(i), '. SELL at price ', str(V[self.time])])
                    self.sellt = np.insert(self.sellt, self.sellt.size, self.timeSeries.Tid + self.time)
                    self.sellp = np.insert(self.sellp, self.sellp.size, currval)
                    # self.timeSeries.Tid = self.timeSeries.Tid + i
                    # self.plotPortfolioAndTrade(strategy, True)
                    self.timeSeries.Pid = self.timeSeries.P_all[:, range(self.time, self.timeSeries.Tid + self.time)]
                    self.timeSeries.Tperf = self.timeSeries.Tperf - self.localTime - 1
                    self.timeSeries.Pperf = self.timeSeries.P_all[:, range(self.timeSeries.Tid + self.time, self.timeSeries.Tid + self.time + self.timeSeries.Tperf)]
                    self.timeSeries.makeZeroMean()
                    self.ret = (self.C[self.time + 1] - self.portfolios.C0) / self.portfolios.C0
                    self.profit = (self.C[self.time + 1] - self.portfolios.C0)

                    # return portfheld, self.V, self.C, self.numtrade, sellt, sellp, buyt, buyp
                    # tvec = [self.mu[i], self.timeSeries.Beta, self.portfolios.xoptbuy, self.V[i], self.x[i], self.buyt[-1], self.sellt[-1]]
                    # tvec.savetxt()
                else:
                    # hold.
                    self.C[self.time + 1] = self.C[self.time]
                    self.x[self.time + 1, :] = self.x[self.time, :]

            self.time += 1
            self.localTime += 1
            empirical_dist, bins = np.histogram(
                self.portfolios.xoptbuy0.dot(self.timeSeries.P_all[:, 20:self.timeSeries.Tid + self.time]), bins=15,
                density=True)
            ou_dist = self.OUfromFokkerPlanck(abs(self.portfolios.weights['greedy'][1]),
                                              self.timeSeries.est_sigma2() / 2, self.time, 0, bins[1:],
                                              self.portfolios.xoptbuy0.dot(self.timeSeries.Pid[:, 20]), self.mu0)
            self.Idivergence(ou_dist, empirical_dist)

        # import matplotlib
        # aa = (np.histogram(self.sellp - self.buyp[0:self.sellp.size], bins=np.arange(50)))
        # df=pd.DataFrame({'lambda': self.portfolios.weights['annealing'][1], 'beta': self.data.Beta, 'mean':self.mu[i], 'stdev': self.eps2 - self.mu[i]})
        # df.hist(bins=50).to_csv("profit_hist_13_10_160_100.csv", mode='a')
        # pd.DataFrame(self.C).to_csv("cash_13_10_160_100.csv", mode='a')
        # pd.DataFrame(self.x).to_csv("weights_13_10_160_100.csv", mode='a')
        # pd.DataFrame(self.timeSeries.Beta).to_csv("Beta_13_10_160_100.csv", mode='a')
        # return portfheld, self.V, self.C, self.numtrade, self.sellt, self.sellp, self.buyt, self.buyp
        # pl.clf()
        # pl.xlabel('Time [Days]')
        # pl.ylabel('Portfolio value [$]')
        # pl.title('Value of portfolio during trading period')
        # pl.plot(range(self.timeSeries.Tid), self.portfolios.xoptbuy0.dot(self.timeSeries.Pid), 'g-')
        # pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid + self.timeSeries.Tperf),
        #         self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf), 'r-')
        # pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid + self.timeSeries.Tperf),
        #         self.portfolios.xoptbuy0.dot(self.timeSeries.generateFromOnline(self.timeSeries.Tperf, 0)[:-1, :]), 'c-')
        if self.portfheld:
            self.profit = self.C[self.time] - self.portfolios.C0 + self.V[self.time-1]
            self.dump(self.mu[self.time], self.C[self.time])
        else:
            self.profit = self.C[self.time] - self.portfolios.C0 + self.V[self.time]
            self.dump(self.mu[self.time], self.C[self.time])

    def process(self):
# def Trading(self, Tperf, eps1, eps2, mu0, xoptbuy, xoptnorm, Pid, P_all, Pperf, Tid, C0, risktaker, n):
        # Check whether to trade immediatel
        lookback_days = 3
        self.timeSeries.parameterEstimation()
        lstmpred = LSTMPrediction.LSTMPrediction(np.concatenate((self.portfolios.xoptbuy0.dot(self.timeSeries.Pid).reshape(
            self.timeSeries.Tid, 1), self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf).reshape(self.timeSeries.Tperf, 1))),self.timeSeries.Tid/(self.timeSeries.Tid+self.timeSeries.Tperf))
        lstmtrainedPortfolio, lstmpredictedPortfolio, lstmpredictedPortfolio2 = lstmpred.prediction(self.timeSeries.Tid/(self.timeSeries.Tid+self.timeSeries.Tperf))
        locmin = argrelextrema(lstmpredictedPortfolio, np.less)
        locmax = argrelextrema(lstmpredictedPortfolio, np.greater)

        pl.clf()
        pl.xlabel('Time [Days]', fontsize=22)
        pl.ylabel('Portfolio value [$]', fontsize=22)
        pl.title('Value of portfolio during trading period', fontsize=26)
        pl.xticks(fontsize=18)
        pl.yticks(fontsize=18)
        pl.plot(range(self.timeSeries.Tid), self.portfolios.xoptbuy0.dot(self.timeSeries.Pid),'g-')
        pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid+self.timeSeries.Tperf),   self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf),'r-')
        pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid+self.timeSeries.Tperf), self.portfolios.xoptbuy0.dot(self.timeSeries.generateNoiselessFromA(self.timeSeries.Tperf, self.timeSeries.Pperf[:, 0])[:-1, :]),'b-')
        pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid+self.timeSeries.Tperf), self.portfolios.xoptbuy0.dot(self.timeSeries.generateFromOnline(self.timeSeries.Tperf, 0)[:-1, :]),'c-')
        pl.plot(range(self.timeSeries.Tid+int(np.floor((1+lookback_days)/2)), self.timeSeries.Tid+int(np.floor((1+lookback_days)/2))+len(lstmpredictedPortfolio)), lstmpredictedPortfolio,'y-')
        # pl.plot(self.timeSeries.Tid+int(np.floor((1+lookback_days)/2))+locmin[0], lstmpredictedPortfolio[locmin],'x')
        # pl.plot(self.timeSeries.Tid+int(np.floor((1+lookback_days)/2))+locmax[0], lstmpredictedPortfolio[locmax],'o')
        pl.plot([0, self.timeSeries.Tid+self.timeSeries.Tperf],[self.eps1, self.eps1], '--')
        pl.plot([0, self.timeSeries.Tid+self.timeSeries.Tperf],[self.eps2, self.eps2], '--')
        pl.legend(['Calibration range, real data','Trading range, real data', 'VAR(1) Offline', 'VAR(1) Online', 'LSTM Online','Buy threshold', 'Sell threshold'], fontsize=15)
        # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])

        fig = pl.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        tickList = '_'.join(self.timeSeries.ticker)
        realdataPath = '.\\Portfolio_N{i}_{s}'.format(i=self.timeSeries.n, s=self.timeSeries.L) + tickList[:60] + '.png'
        pl.savefig(realdataPath)

        pd.DataFrame(self.timeSeries.tr_P_all).to_csv(self.timeSeriesFileName, mode='a')

        # self.mu0 = np.mean(lstmpredictedPortfolio)
        # self.eps1 = self.mu0 - 0.7*np.std(lstmpredictedPortfolio)
        # self.eps2 = self.mu0 + 0.7*np.std(lstmpredictedPortfolio)
        self.tradingRange = self.eps2 -self.eps1

        # pd.DataFrame(np.array([self.timeSeries.n, self.timeSeries.L, self.timeSeries.simulated, self.timeSeries.useShift, self.timeSeries.meanEstimationError()])).to_csv(self.meanErrorFileName)
        if self.eps1 > self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, self.timeSeries.Tid - 1]):
            V0 = self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, self.timeSeries.Tid - 1])
            # print('0. BUY at price ', str(V0))
            self.portfheld = True
            self.buyt = np.insert(self.buyt, self.buyt.size, self.timeSeries.Tid)
            self.buyp = np.insert(self.buyp, self.buyp.size, V0)
            self.C[0] = self.portfolios.C0 - V0
            self.x[0, :] = self.portfolios.xoptbuy
            self.numtrade += 1
        else:
            self.C[0] = self.portfolios.C0
            self.x[0, :] = np.zeros(self.timeSeries.n)
            # V0 = 0

        i = 0
        self.time = 0

        while i < self.timeSeries.Tperf:
            if self.selled:
                # self.dump(self.mu[i-1], self.C[i-1])
                self.timeSeries.parameterEstimation()
                self.portfolios.selection()
                self.selled = False
                self.mu[i] = self.portfolios.mu0
                self.eps1 = self.mu[i] - np.std(self.portfolios.xoptbuy0.dot(self.timeSeries.Pid[:, 20:]))
                self.eps2 = self.mu[i] + np.std(self.portfolios.xoptbuy0.dot(self.timeSeries.Pid[:, 20:]))
                self.tradingRange = self.eps2 - self.eps1
        # for i in range(Tperf):
            # compute mu in terms of orig portfolio
            # mu[i] = estMu.est_mu_linreg(xoptbuy0.dot(P_all[:, range(Tid + i)]))
            self.portfolios.est_mu_linreg()
            # self.mu[i] = self.portfolios.mu0
            self.mu[i] = np.mean(self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, -30:-1]))
            currval = self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf[:, i])
            # df = pd.DataFrame({ 'lambda': self.portfolios.weights['annealing'][1]}, index=[0])
            # df.to_csv("eigenval_13_10_160_100.csv", mode='a')
            if not self.portfheld:
                # check if we should buy
                if (self.eps1 > currval) & (self.timeSeries.Tperf - (self.time - self.timeSeries.Tid) > 20):
                    self.portfheld = True
                    if self.risktaker:
                        # Case 1: wager all our money
                        self.portfolios.xoptsp = self.C[i] * self.portfolios.xoptnorm

                    # Case 2: we only spend C0, so do not recompute xoptsp
                    self.portfolios.xoptbuy = np.floor(self.portfolios.xoptsp / self.timeSeries.tr_Pperf[:, i])
                    self.V[i] = self.portfolios.xoptbuy.dot(self.timeSeries.Pperf[:, i])
                    self.C[i] = self.C[i] - self.V[i]
                    self.C[i + 1] = self.C[i]
                    self.x[i + 1, :] = self.portfolios.xoptbuy
                    self.numtrade += 1
                    # print([str(i), '. BUY at price ', str(V[i])])
                    self.buyt = np.insert(self.buyt, self.buyt.size, self.timeSeries.Tid + i)
                    self.buyp = np.insert(self.buyp, self.buyp.size, currval)
                else:
                    # keep money
                    self.C[i + 1] = self.C[i]
                    self.V[i] = 0
                    self.x[i + 1, :] = self.x[i, :]

            else:
                # compute portf value at new price
                self.V[i] = self.x[i, :].dot(self.timeSeries.Pperf[:, i])
                # check if we should sell
                if (self.eps2 < currval) |\
                      ((currval - self.buyp[-1] < 0.5 * self.tradingRange) &
                       (self.timeSeries.Tperf - (self.buyt[-1] - self.timeSeries.Tid) < 20)):
                    self.portfheld = False
                    self.selled = True
                    self.C[i + 1] = self.C[i] + self.x[i, :].dot(self.timeSeries.Pperf[:, i])
                    self.V[i + 1] = 0
                    self.x[i + 1, :] = np.zeros(self.timeSeries.n)
                    self.numtrade += 1
                    # print([str(i), '. SELL at price ', str(V[i])])
                    self.sellt = np.insert(self.sellt, self.sellt.size, self.timeSeries.Tid + i)
                    self.sellp = np.insert(self.sellp, self.sellp.size, currval)
                    # self.timeSeries.Tid = self.timeSeries.Tid + i
                    self.timeSeries.Pid = self.timeSeries.P_all[:, range(i, self.timeSeries.Tid + i)]
                    self.timeSeries.makeZeroMean()
                    self.ret = (self.C[i + 1] - self.portfolios.C0) / self.portfolios.C0
                    self.profit = (self.C[i + 1] - self.portfolios.C0)
                    # self.timeSeries.Tperf = self.timeSeries.Tperf - i
                    # self.timeSeries.Pperf = self.timeSeries.P_all[:, range(self.timeSeries.Tid + i, self.timeSeries.Tid + self.timeSeries.Tperf)]
                    # return portfheld, self.V, self.C, self.numtrade, sellt, sellp, buyt, buyp
                    # tvec = [self.mu[i], self.timeSeries.Beta, self.portfolios.xoptbuy, self.V[i], self.x[i], self.buyt[-1], self.sellt[-1]]
                    # tvec.savetxt()
                else:
                    # hold.
                    self.C[i + 1] = self.C[i]
                    self.x[i + 1, :] = self.x[i, :]

            self.time += 1
            i += 1
            empirical_dist, bins = np.histogram(self.portfolios.xoptbuy0.dot(self.timeSeries.P_all[:, 20:self.timeSeries.Tid+i]), bins=15, density=True)
            # ou_dist = self.OUfromFokkerPlanck(abs(self.portfolios.weights['greedy'][1]), self.timeSeries.est_sigma2()/2, i, 0, bins[1:], self.portfolios.xoptbuy0.dot(self.timeSeries.Pid[:, 20]), self.mu0)
            # self.Idivergence(ou_dist, empirical_dist)


    # import matplotlib
        # aa = (np.histogram(self.sellp - self.buyp[0:self.sellp.size], bins=np.arange(50)))
        # df=pd.DataFrame({'lambda': self.portfolios.weights['annealing'][1], 'beta': self.data.Beta, 'mean':self.mu[i], 'stdev': self.eps2 - self.mu[i]})
        # df.hist(bins=50).to_csv("profit_hist_13_10_160_100.csv", mode='a')
        # pd.DataFrame(self.C).to_csv("cash_13_10_160_100.csv", mode='a')
        # pd.DataFrame(self.x).to_csv("weights_13_10_160_100.csv", mode='a')
        # pd.DataFrame(self.timeSeries.Beta).to_csv("Beta_13_10_160_100.csv", mode='a')
        # return portfheld, self.V, self.C, self.numtrade, self.sellt, self.sellp, self.buyt, self.buyp
        if self.portfheld:
            self.profit = self.C[-1] - self.portfolios.C0 + self.V[-2]
            self.dump(self.mu[i], self.C[i])
        else:
            self.profit = self.C[-1] - self.portfolios.C0 + self.V[-1]
            self.dump(self.mu[i], self.C[i])


    def process2(self):
        lookback_days = 30
        self.timeSeries.parameterEstimation()
        # lstmpred = LSTMPrediction.LSTMPrediction(
        #     np.concatenate((self.portfolios.xoptbuy0.dot(self.timeSeries.Pid).reshape(
        #         self.timeSeries.Tid, 1),
        #                     self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf).reshape(self.timeSeries.Tperf, 1))),
        #     self.timeSeries.Tid / (self.timeSeries.Tid + self.timeSeries.Tperf), look_back=lookback_days, numofLSTMLayers=4)
        # lstmtrainedPortfolio, lstmpredictedPortfolio, lstmpredictedPortfolio2 = lstmpred.prediction(
        #     self.timeSeries.Tid / (self.timeSeries.Tid + self.timeSeries.Tperf))
        lstmpred = LuongAttention.NLPwithLSTM(np.concatenate((self.portfolios.xoptbuy0.dot(self.timeSeries.Pid).reshape(self.timeSeries.Tid, 1))))
        lstmtrainedPortfolio = lstmpred.predict()
        locmin = argrelextrema(lstmpredictedPortfolio, np.less)
        locmax = argrelextrema(lstmpredictedPortfolio, np.greater)
        pl.clf()
        pl.xlabel('Time [Days]')
        pl.ylabel('Portfolio value [$]')
        pl.title('Value of portfolio during trading period')
        pl.plot(range(self.timeSeries.Tid), self.portfolios.xoptbuy0.dot(self.timeSeries.Pid), 'g-')
        pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid + self.timeSeries.Tperf),
                self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf), 'r-')
        pl.plot(range(self.timeSeries.Tid, self.timeSeries.Tid + self.timeSeries.Tperf),
                self.portfolios.xoptbuy0.dot(self.timeSeries.generateNoiselessFromA(self.timeSeries.Tperf)[:-1, :]),
                'b-')
        # pl.plot(range(self.timeSeries.Tid + int(np.floor((1 + lookback_days) / 2)),
        #               self.timeSeries.Tid + self.timeSeries.Tperf - int(np.ceil((1 + lookback_days) / 2))),
        #         lstmpredictedPortfolio, 'y-')
        # pl.plot(self.timeSeries.Tid + int(np.floor((1 + lookback_days) / 2)) + locmin[0],
        #         lstmpredictedPortfolio[locmin], 'x')
        # pl.plot(self.timeSeries.Tid + int(np.floor((1 + lookback_days) / 2)) + locmax[0],
        #         lstmpredictedPortfolio[locmax], 'o')
        pl.plot(range(self.timeSeries.Tid + lookback_days - 1,
                      self.timeSeries.Tid + len(lstmpredictedPortfolio) + lookback_days-1),
                lstmpredictedPortfolio, 'y-')
        pl.plot(range(self.timeSeries.Tid + lookback_days - 1,
                      self.timeSeries.Tid + len(lstmpredictedPortfolio2) + lookback_days-1),
                lstmpredictedPortfolio2, 'y-')
        pl.plot(self.timeSeries.Tid + lookback_days - 1 + locmin[0],
                lstmpredictedPortfolio[locmin], 'x')
        pl.plot(self.timeSeries.Tid + lookback_days - 1 + locmax[0],
                lstmpredictedPortfolio[locmax], 'o')
        # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])

        fig = pl.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        tickList = '_'.join(self.timeSeries.ticker)
        realdataPath = '.\\Portfolio_N{i}_{s}'.format(i=self.timeSeries.n, s=self.timeSeries.L) + tickList[:60] + '.png'
        pl.savefig(realdataPath)

        pd.DataFrame(self.timeSeries.tr_P_all).to_csv(self.timeSeriesFileName, mode='a')

        self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, self.timeSeries.Tid - 1])

        if lstmpredictedPortfolio[locmin[0][0]] > self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, self.timeSeries.Tid - 1]):
            V0 = self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, self.timeSeries.Tid - 1])
            # print('0. BUY at price ', str(V0))
            self.portfheld = True
            self.buyt = np.insert(self.buyt, self.buyt.size, self.timeSeries.Tid)
            self.buyp = np.insert(self.buyp, self.buyp.size, V0)
            self.C[0] = self.portfolios.C0 - V0
            self.x[0, :] = self.portfolios.xoptbuy
            self.numtrade += 1
        else:
            self.C[0] = self.portfolios.C0
            self.x[0, :] = np.zeros(self.timeSeries.n)
             # V0 = 0

        i = 0

        while i < self.timeSeries.Tperf:
            if self.selled:
                self.timeSeries.parameterEstimation()
                self.portfolios.selection()
                self.selled = False
            currval = self.portfolios.xoptbuy0.dot(self.timeSeries.Pperf[:, i])
            if not self.portfheld:
                # check if we should buy
                if (i - int(np.floor((1 + lookback_days) / 2)) in locmin[0]) & (any(i - int(np.floor((1 + lookback_days) / 2)) < num for num in locmax[0])):
                    self.portfheld = True
                    if self.risktaker:
                        # Case 1: wager all our money
                        self.portfolios.xoptsp = self.C[i] * self.portfolios.xoptnorm

                    # Case 2: we only spend C0, so do not recompute xoptsp
                    self.portfolios.xoptbuy = np.floor(self.portfolios.xoptsp / self.timeSeries.tr_Pperf[:, i])
                    self.V[i] = self.portfolios.xoptbuy.dot(self.timeSeries.Pperf[:, i])
                    self.C[i] = self.C[i] - self.V[i]
                    self.C[i + 1] = self.C[i]
                    self.x[i + 1, :] = self.portfolios.xoptbuy
                    self.numtrade += 1
                    # print([str(i), '. BUY at price ', str(V[i])])
                    self.buyt = np.insert(self.buyt, self.buyt.size, self.timeSeries.Tid + i)
                    self.buyp = np.insert(self.buyp, self.buyp.size, currval)
                else:
                    # keep money
                    self.C[i + 1] = self.C[i]
                    self.V[i] = 0
                    self.x[i + 1, :] = self.x[i, :]

            else:
                # compute portf value at new price
                self.V[i] = self.x[i, :].dot(self.timeSeries.Pperf[:, i])
                # check if we should sell
                if i - int(np.floor((1 + lookback_days) / 2)) in locmax[0]:
                    self.portfheld = False
                    self.selled = True
                    self.C[i + 1] = self.C[i] + self.x[i, :].dot(self.timeSeries.Pperf[:, i])
                    self.V[i + 1] = 0
                    self.x[i + 1, :] = np.zeros(self.timeSeries.n)
                    self.numtrade += 1
                    # print([str(i), '. SELL at price ', str(V[i])])
                    self.sellt = np.insert(self.sellt, self.sellt.size, self.timeSeries.Tid + i)
                    self.sellp = np.insert(self.sellp, self.sellp.size, currval)
                    # self.timeSeries.Tid = self.timeSeries.Tid + i
                    self.timeSeries.Pid = self.timeSeries.P_all[:, range(i, self.timeSeries.Tid + i)]
                    self.timeSeries.makeZeroMean()
                    self.ret = (self.C[i + 1] - self.portfolios.C0) / self.portfolios.C0
                    self.profit = (self.C[i + 1] - self.portfolios.C0)
                    # self.timeSeries.Tperf = self.timeSeries.Tperf - i
                    # self.timeSeries.Pperf = self.timeSeries.P_all[:, range(self.timeSeries.Tid + i, self.timeSeries.Tid + self.timeSeries.Tperf)]
                    # return portfheld, self.V, self.C, self.numtrade, sellt, sellp, buyt, buyp
                    # tvec = [self.mu[i], self.timeSeries.Beta, self.portfolios.xoptbuy, self.V[i], self.x[i], self.buyt[-1], self.sellt[-1]]
                    # tvec.savetxt()
                else:
                    # hold.
                    self.C[i + 1] = self.C[i]
                    self.x[i + 1, :] = self.x[i, :]

            i += 1
            empirical_dist, bins = np.histogram(
                self.portfolios.xoptbuy0.dot(self.timeSeries.P_all[:, 20:self.timeSeries.Tid + i]), bins=15,
                density=True)
            ou_dist = self.OUfromFokkerPlanck(abs(self.portfolios.weights['greedy'][1]),
                                              self.timeSeries.est_sigma2() / 2, i, 0, bins[1:],
                                              self.portfolios.xoptbuy0.dot(self.timeSeries.Pid[:, 20]), self.mu0)
            self.Idivergence(ou_dist, empirical_dist)

        # import matplotlib
        # aa = (np.histogram(self.sellp - self.buyp[0:self.sellp.size], bins=np.arange(50)))
        # df=pd.DataFrame({'lambda': self.portfolios.weights['annealing'][1], 'beta': self.data.Beta, 'mean':self.mu[i], 'stdev': self.eps2 - self.mu[i]})
        # df.hist(bins=50).to_csv("profit_hist_13_10_160_100.csv", mode='a')
        # pd.DataFrame(self.C).to_csv("cash_13_10_160_100.csv", mode='a')
        # pd.DataFrame(self.x).to_csv("weights_13_10_160_100.csv", mode='a')
        # pd.DataFrame(self.timeSeries.Beta).to_csv("Beta_13_10_160_100.csv", mode='a')
        # return portfheld, self.V, self.C, self.numtrade, self.sellt, self.sellp, self.buyt, self.buyp
        if self.portfheld:
            self.profit = self.C[-1] - self.portfolios.C0 + self.V[-2]
            self.dump(self.mu[i], self.C[i])
        else:
            self.profit = self.C[-1] - self.portfolios.C0 + self.V[-1]
            self.dump(self.mu[i], self.C[i])

    def getNext(self):
        return self.timeSeries.Pperf

    def dump(self, mu, cash):
        bins = 30
        tempArray = np.zeros(18+2*bins+1)
        tempArray[0] = self.portfheld
        tempArray[1] = self.profit
        tempArray[2] = self.numtrade
        tempArray[3] = self.portfolios.weights['exhaustive'][1]
        tempArray[4] = self.timeSeries.Beta
        tempArray[5] = mu
        tempArray[6] = self.eps2 - mu
        tempArray[7] = cash
        tempArray[8] = self.timeSeries.L
        tempArray[9] = self.timeSeries.n
        tempArray[10] = self.timeSeries.Tid
        tempArray[11] = self.timeSeries.simulated
        tempArray[12] = self.timeSeries.useShift
        tempArray[13] = self.timeSeries.useShift
        # tempArray[13] = self.timeSeries.meanEstimationError()
        ix = 15
        tempArray[14:ix] = self.buyt[-1] if len(self.buyt) > 0 else 0
        tempArray[ix:ix+int(self.numtrade/2)] = self.buyp[-1] if len(self.buyt) > 0 else 0
        ix += 1
        tempArray[ix:ix + int(self.numtrade/2)] = self.sellt[-1] if len(self.sellt) > 0 else 0
        ix += 1
        tempArray[ix:ix + int(self.numtrade/2)] = self.sellp[-1] if len(self.sellt) > 0 else 0
        ix += 1
        aa = np.histogram(self.portfolios.xoptbuy.dot(self.timeSeries.Pid[:, 20:]), bins)
        tempArray[ix:ix + bins] = aa[0]
        ix += bins
        tempArray[ix:ix + bins + 1] = aa[1]
        with open(self.cashFileName, 'a') as outfile:
            np.savetxt(outfile, [tempArray], delimiter='\t', fmt='%f')
            # np.savetxt(outfile, np.transpose(tempArray), newline=" ", fmt='%f')
            # np.savetxt(outfile,  newline="\n")
