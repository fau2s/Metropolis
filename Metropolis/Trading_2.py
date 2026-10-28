# from Metropolis import data
# from Metropolis import Trading
import data
import Trading
import numpy as np
import argparse
# dummyTimeSeries = data.Data()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action="store_true", default=False)
    parser.add_argument('-b', action="store", dest="b")
    parser.add_argument('-c', action="store", dest="c", type=int)
    args = parser.parse_args()

    beta = np.array([], dtype=float)
    lamb = np.array([], dtype=float)
    Return = np.array([], dtype=float)
    profit = np.array([], dtype=float)

    population = [10, 13, 16, 19]
    # population = [19]0

    # for pop in population:
    #     for spar in range(9, pop-2):
    #         for id in range(150, 200, 50):
    #             for i in range(200):
    #                 dummyTrade = Trading.Trading(n=pop, spars=spar, simulated=False, idTimeWindow=id, tradingTimeWindow=100, useShift=True, n_diff=0)
    #                 dummyTrade.process()
    #                 lamb = np.insert(lamb, lamb.size, dummyTrade.portfolios.weights['annealing'][1])
    #                 Return = np.insert(Return, Return.size, dummyTrade.ret)
    #                 profit = np.insert(profit, profit.size, dummyTrade.profit)
    #                 print(i)

    # print(dummyTrade.timeSeries.P_all)
    for spar in range(17, 30, 5):
        for i in range(2100, 2300, 100):
            for id in range(1900, 2000, 200):
                dummyTrade = Trading.Trading(n=430, spars=spar, simulated=False, idTimeWindow=id, tradingStart=i, tradingTimeWindow=50, useShift=True, n_diff=0, model='log', sigma=1e-1)
                # dummyTrade.processOnOffline("OnlineLSTM")
                # dummyTrade = Trading.Trading(n=18, spars=13, simulated=False, idTimeWindow=id, tradingTimeWindow=100, useShift=False, n_diff=0)
                # dummyTrade.process()
                dummyTrade.calibrating()
                lamb = np.insert(lamb, lamb.size, dummyTrade.portfolios.weights['annealing'][1])
                Return = np.insert(Return, Return.size, dummyTrade.ret)
                profit = np.insert(profit, profit.size, dummyTrade.profit)
                print(id)


if __name__ == '__main__':
    main()