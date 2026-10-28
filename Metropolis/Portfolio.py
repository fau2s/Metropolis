import numpy as np
import pandas as pd
# from numpy.ma.bench import timer
from timeit import default_timer as timer

# from Metropolis import data
# from Metropolis import annealing as ann
import annealing as ann
import data
import annealing as ann

from scipy.optimize import least_squares
# from Metropolis import ExtremeLearningMachine as eml


class Portfolio:
    def __init__(self, timeSeries, initialCash=10000):
        self.optimizationTypes = ['exhaustive', 'truncation', 'greedy', 'annealing']
        self.timeSeries = timeSeries
        self.C0 = initialCash
        self.weights = {}
        # self.Win = np.load('E:\PhD\Metropolis\InputWeights_{0}'.format(self.timeSeries.n)+'.npy')
        # self.Wout = np.load('E:\PhD\Metropolis\OutputWeights_{0}'.format(self.timeSeries.n) + '.npy')
        self.EMLtestFile = "EMLTestInRun_{0}_{1}.csv".format(self.timeSeries.n, self.timeSeries.L)
        for meth in self.optimizationTypes:
            self.weights[meth] = {'weights': np.zeros(self.timeSeries.L), 'eigenvalue': 0}
        # self.weights['exhaustive'] = self.timeSeries.solve_exhaust()
        self.selection()
        self.xoptbuy0 = self.xoptbuy.copy()
        # self.weights['truncation'] = self.timeSeries.solve_trunc()
        # self.weights['greedy'] = self.timeSeries.solve_greedy()
        # an = ann.anneal(self.weights['greedy'][0], self.weights['greedy'][0], self.timeSeries.A, self.timeSeries.G,
        #                 self.timeSeries.L)
        # self.weights['annealing'] = an.process()


    def selection(self):
        self.sparse_meanrev_portfselect()
        # runtime[rep,meth]=toc
        # lam[rep, meth - 1] = lamb
        # if self.meth == 1:
            # wexh = xopt  # save wexh for annealing.py comparison

        # Allow shorting, but display a warning message
        if np.sum(self.weights['annealing'][0] >= 0) != self.timeSeries.A.shape[0]:
            print('Negative element in xopt - will need to do shorting')

        # Scale portfolio so that we spend exactly C0
        self.xoptnorm = self.weights['annealing'][0] / sum(abs(self.weights['annealing'][0]))  # normalize xopt so that sum = 1
        self.xoptsp = self.C0 * self.xoptnorm  # amount of money to spend on each asset
        self.xoptbuy = np.floor(self.xoptsp / self.timeSeries.tr_Pid[:, self.timeSeries.Tid - 1])  # number of contracts to buy of each asset
        # Compute historical value of xoptbuy portfolio during Tid
        self.histportfval = np.matmul(self.xoptbuy, self.timeSeries.tr_Pid)

        self.diffHistPortfVal = np.diff(self.histportfval)

        # from matplotlib import pyplot as plt
        # plt.plot(self.histportfval[:-1], self.diffHistPortfVal, 'x')
        # plt.show()

        # Estimate the mu associatmued with xoptbuy
        self.est_mu_linreg()

        self.xoptbuy0 = self.xoptbuy.copy()
        # return xopt, lamb, xoptbuy, xoptnorm, mu0

    def sparse_meanrev_portfselect(self):
        # n = A.shape
        # self.weights['exhaustive'] = self.timeSeries.solve_exhaust()
        # self.weights['truncation'] = self.timeSeries.solve_trunc()
        self.weights['greedy'] = self.timeSeries.solve_greedy()
        self.weights['exhaustive'] = self.weights['greedy']
        self.weights['annealing'] = self.weights['greedy']
        an = ann.anneal(self.weights['annealing'][0].copy(), self.weights['annealing'][0].copy(), self.timeSeries.A, self.timeSeries.G, self.timeSeries.L)
        self.weights['annealing'] = an.process(False)
    # if np.argmax(eml.predict(np.append(self.weights['greedy'][1],(np.sum(self.weights['greedy'][0] != 0))), self.Win, self.Wout)) == 1:
        # # if np.argmax(eml.predict(np.append(self.weights['greedy'][0], (np.sum(self.weights['greedy'][0] != 0))),self.Win, self.Wout)) == 1:
        #     if np.abs(self.weights['greedy'][1] - self.weights['exhaustive'][1]) < 0.008:
        #         pd.DataFrame({'a': [self.weights['exhaustive'][1]], 'b': [self.weights['greedy'][1]],
        #                       'c': [self.weights['exhaustive'][1]-self.weights['greedy'][1]],
        #                       'd': [1], 'e': [0], 'f': [0], 'g': [0]}).to_csv(self.EMLtestFile, header=False, index=False, mode='a')
        #         print('Nem kellene!')
        #     else:
        #         pd.DataFrame({'a': [self.weights['exhaustive'][1]], 'b': [self.weights['greedy'][1]],
        #                       'c': [self.weights['exhaustive'][1]-self.weights['greedy'][1]],
        #                       'd': [0], 'e': [1], 'f': [0], 'g': [0]}).to_csv(
        #             self.EMLtestFile, header=False, index=False, mode='a')
        #         print("Kellett!")
        #     an = ann.anneal(self.weights['greedy'][0], self.weights['greedy'][0], self.timeSeries.A, self.timeSeries.G, self.timeSeries.L)
        #     print('newtimer')
        #     start = timer()
        #     newWeight = an.process(False)
        #     print(timer() - start)
        #     print('oldtimer')
        #     start = timer()
        #     oldWeight = an.process(True)
        #     print(timer()-start)
        #     self.weights['annealing'] = an.process(False)
        #     pd.DataFrame({'exhaustive': [self.weights['exhaustive'][1]], 'old greedy': [oldWeight[1]],
        #                   'new greedy': [self.weights['annealing'][1]]}).to_csv("OldGraphTest.csv", header=False,
        #                                                                         index=False, mode='a')
        # else:
        #     if np.abs(self.weights['greedy'][1] - self.weights['exhaustive'][1]) > 0.008:
        #         pd.DataFrame({'a': [self.weights['exhaustive'][1]], 'b': [self.weights['greedy'][1]],
        #                       'c': [self.weights['exhaustive'][1]-self.weights['greedy'][1]],
        #                       'd': [0], 'e': [0], 'f': [1], 'g': [0]}).to_csv(
        #             self.EMLtestFile, header=False, index=False, mode='a')
        #         print('Pedig kellene!')
        #     else:
        #         pd.DataFrame({'a': [self.weights['exhaustive'][1]], 'b': [self.weights['greedy'][1]],
        #                       'c': [self.weights['exhaustive'][1]-self.weights['greedy'][1]],
        #                       'd': [0], 'e': [0], 'f': [0], 'g': [1]}).to_csv(
        #             self.EMLtestFile, header=False, index=False,  mode='a')
        #         print("Nem is kellett!")
        # if method == 1:  # Exhaustive
        #     [portf, lamb, num] = solve_exhaust(A, G, L)
        #     return [portf, lamb]
        # else:
        #     if method == 4:  # Truncation
        #         [portf, lamb] = solve_trunc(A, G, L)
        #         return [portf, lamb]
        #     else:
        #         # Greedy search
        #         [portf, lamb] = solve_greedy(A, G, L)
        #         if method == 2:  # Greedy
        #             return [portf, lamb]

        # Simulated annealing.py
        # start from greedy solution
        # initw = 1000 * portf

        # Simulated annealing.py using new neigh function
        # start from greedy solution!!

        # opts = struct('Generator', @anneal_neigh2, 'MaxTries',3000,'MaxConsRej', 20000)
        # an = ann.anneal(initw, 1000 * portf, A, G, L)
        # [wann3, dmax] = an.process()
        # [wann3, dmax] = ann.simulatedAnnealing(1, portf, 1000, A, G)
        # annealing.py minimizes, so store -dmax as the max.
        # dsimann3 = -dmax
        # if -self.weights['annealing'].lam > self.lamb['greedy']:
        #     lamb = dsimann3
        #     portf = wann3
        #
        # return [portf, lamb]

    def est_mu_linreg(self):
        if (np.isnan(np.sum(self.histportfval))):
            mu = np.NaN
            return mu

        x0 = np.ones(2)
        # coeff = robustfit(rho[2:rho.size], rho[1:rho.size-1])
        coeff = least_squares(self.fun, x0, loss='soft_l1', f_scale=0.1, args=(self.histportfval[20:self.histportfval.size - 1], self.histportfval[19:self.histportfval.size - 2]))
        self.mu0 = coeff.x[1] / (1 - coeff.x[0])

    def fun(self, x, rho, y):
        return x[0] * rho + x[1] - y