import numpy as np
# from Metropolis import linAlg as la
import linAlg as la
import pandas as pd
from collections import defaultdict
import quandl as qa
from yahoofinancials import YahooFinancials
from statsmodels.stats import correlation_tools as corrtools
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
import math

class Data:
    def __init__(self, n=8, L=7, sigma=0.01, Tid=300, Tperf=200, Tstart=300, simulated=True, useShift=True, n_diff=0, tickers='0', model='regular', VARparameter=4):
        self.maxlam = 0
        self.loadfile = 1
        self.maxlamLimit = 0.05
        self.simulated = simulated

        self.model = model
        self.useShift = useShift

        self.n = n  # Total number of assets to choose from
        self.L = L   # Max number of assets allowed in sparse portfolio
        self.Tid = Tid  # Size of identification window
        self.Tperf = Tperf  # Size of trading window

        self.VARparameter = VARparameter

        self.r = 0  # daily compounded risk free interest rate

        # Define internal model parameters
        self.sigma = sigma  # size of the noise in generated process
        self.TG = 50  # size of time window used for estimating G

        self.A = np.zeros((self.n, self.n))
        self.G = np.zeros((self.n, self.n))

        self.cshift = 0
        self.est_cshift = 0
        self.calib_cshift = np.zeros((self.n, 1))

        self.correlationType = True

        self.ticker = tickers

        if self.simulated:
            self.generateSimulated()
        else:
            self.readFromFile()

        self.dataLength = self.P_all.shape[1]

        self.Tstart = np.min((np.max((Tstart, 0)), self.dataLength))

        if self.Tstart < self.Tid:
            self.Tid = self.Tstart
        elif self.dataLength - self.Tstart < self.Tperf:
            self.Tperf = self.dataLength - self.Tstart
        elif self.Tid + self.Tperf > self.dataLength:
            self.Tperf = self.dataLength - self.Tid
        elif self.Tstart > self.dataLength:
            pass

        self.setIdandPerf()

        # self.estA()
        self.estAVarP(self.VARparameter)
        # self.est_G_lyap_numeric_corrnoise()
        self.G = np.cov(self.Pid)
        [x, self.maxlam, _] = self.solve_greedy()

        self.tr()
        self.selectModel(1)

        self.makeZeroMean()

    def readFromFile(self):
        tickerList = pd.read_csv('D:\\PhD\\Metropolis\\SP500TickerList.txt')
        cryptoList = pd.read_csv('D:\\PhD\\Metropolis\\cryptoList.txt')
        self.P_all = np.ones([self.n, 1])
        tmp = defaultdict(lambda: None)
        tmpVolume = defaultdict(lambda: None)
        isGood = True

        while isGood:
            # self.ticker = np.random.choice(tickerList[tickerList['Size'] == 2862]['Ticker'], self.n, replace=False)
            self.ticker = np.random.choice(cryptoList['Ticker'], 7, replace=False)
            # if any(ticker in ["ALGN.csv" , "CARR.csv" , "CTVA.csv" , "DOW.csv" ,"FOX.csv", "FOXA.csv" , "LUMN.csv" , "OTIS.csv", "VIAC.csv" , "VNT.csv"] for ticker in self.ticker):
            #     continue
            for ix, data in enumerate(self.ticker):
                ixStr = str(ix)
                tmp[ixStr] = pd.read_csv('D:\\PhD\\Metropolis\\RealData9\\' + data, header=None)
                # tmpVolume[ixStr] = pd.read_csv('D:\\PhD\\Metropolis\\RealData6\\' + data, header=None)
                mask = np.isnan(tmp[ixStr][1])
                # tmp.loc[ixStr, 1] = np.where(mask, np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), tmp.loc[ixStr, 1][~mask]), tmp.loc[ixStr, 1])
                tmp[ixStr][1][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), tmp[ixStr][1][~mask])
                # tmpVolume[ixStr][1][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), tmpVolume[ixStr][1][~mask])
                # tmpVolume.loc[ixStr, 1] = np.where(
                #     mask,
                #     np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),tmpVolume.loc[ixStr, 1][~mask]),
                #     tmpVolume.loc[ixStr, 1]
                # )
                if not ix:
                    self.commonDates = tmp[ixStr][0]
                self.commonDates = np.intersect1d(self.commonDates, tmp[ixStr][0])

                if np.std(tmp[ixStr][tmp[ixStr][0].isin(self.commonDates)][1]) == 0:
                    break

            else:
                isGood = False

        self.P_all.resize((self.n, len(self.commonDates)), refcheck=False)
        # self.P_all = np.resize(self.P_all, (self.n, len(commonDates)))
        for ix in range(len(self.ticker)):
            ixStr = str(ix)
            # self.P_all[ix, :] = (
            #         tmp[ixStr][tmp[ixStr][0].isin(self.commonDates)][1] *
            #         tmpVolume[ixStr][tmpVolume[ixStr][0].isin(self.commonDates)][1])
            self.P_all[ix, :] = tmp[ixStr][tmp[ixStr][0].isin(self.commonDates)][1]

        self.stocks = tmp
        # for ix, data in enumerate(self.ticker):
        #     print(data)
        #     # print(len([close['close'] for i, close in enumerate(price_data[data]['prices'])]))
        #     if ('prices' in price_data[data].keys()):
        #         print(len([close['close'] for i, close in enumerate(price_data[data]['prices']) if 'close' in close.keys()]))
        #         if (len(price_data[data]['prices']) == 775):
        #             self.P_all[ix, :] = [close['close'] for i, close in enumerate(price_data[1data]['prices'])]


    def generateSimulated(self):
        # while self.maxlam < self.maxlamLimit:  # set limit here
        [self.A_true, self.P_all, self.cshift] = self.generateVAR1()
            # self.Pid = self.P_all[:, self.Tstart-self.Tid, self.Tstart]
            # self.Pperf = self.P_all[:, range(self.Tstart, self.Tperf+self.Tstart)]
            # self.estA()
            # self.est_G_lyap_numeric_corrnoise()
            # [x, self.maxlam, _] = self.solve_greedy()
            # print(maxlam)


    def generateVAR1(self):
        # eg. n=15; t=1000; sigma=0.1;
        #############################
        ## Set up sample A matrix
        #############################
        # generate U*V' orthonormal basis
        done = False
        # Try until we find A and c such that the generated sequence is all
        # positive
        kk = 1
        t = self.Tid + self.Tperf
        while not done:
            kk += 1
            # print(kk, '\n')
            # M = np.random.randn(n, n)
            [U, S, V] = np.linalg.svd(np.random.randn(self.n, self.n))
            # eigenvalue diagonal matrix
            D = np.diagflat(np.random.rand(self.n, 1))
            # orthogonal matrix
            O = np.matmul(U, np.transpose(V))
            A = np.matmul(D, np.linalg.inv(O))
            A = np.matmul(O, A)
            # A = np.matmul(U, np.matmul(np.transpose(V), np.matmul(D, np.linalg.inv(np.matmul(U, np.transpose(V))))))
            s = np.random.rand(self.n, t) * 100

            # generate s matrix from A plus noise11
            for c in np.arange(0, 30, 0.2):
                for i in range(t - 1):
                    # add constant at each step to stay positive
                    s[:, i + 1] = A.dot(s[:, i]) + np.random.normal(0, self.sigma, self.n) + c * np.ones(self.n)
                    # check to ensure we stay positive
                    if (np.sum(s[:, i + 1] > 0) != self.n):
                        break
                if i == t - 2:
                    done = True
                    break
        return A, s, c

    def generateNoiselessFrom_A(self, length, start, p):
        size = self.Pid.shape[0]
        # varModel = VAR(self.Pid.transpose())
        # result = varModel.fit(p)
        # fittedResult = varModel.predict(self.Pid.transpose(),lags=p)
        i = p
        noiselessTS = np.zeros((size, length))
        noiselessTS[:, :p] = start.copy()
        while i < length:
            for k in range(p):
                noiselessTS[:, i] += self.result.coefs[k].dot(noiselessTS[:, i - 1 - k])
            noiselessTS[:, i] += self.result.coefs_exog.reshape(size)
            i += 1
        return noiselessTS

    def generateNoiselessFromA(self, length, start):
        i = 1
        if self.useShift:
            noiselessTS = np.ones((self.P_all.shape[0] + 1, length))
            noiselessTS[:-1, 0] = start
        else:
            noiselessTS = np.ones((self.P_all.shape[0], length))
            noiselessTS[:, 0] = start
        while i < length:
            noiselessTS[:, i] = self.Aext.dot(noiselessTS[:, i-1])
            i += 1
        return noiselessTS

    def generateFromOnline(self, length, start):
        i = 1
        if self.useShift:
            # noiselessTS = np.ones((self.P_all.shape[0] + 1, length))
            # noiselessTS[:-1, 0] = self.Pperf[:, 0]
            noiselessTS = np.ones((self.P_all.shape[0], length))
            noiselessTS[:, 0] = self.Pperf[:, 0].copy()
        else:
            noiselessTS = np.ones((self.P_all.shape[0], length))
            noiselessTS[:, 0] = self.Pperf[:, 0].copy()
        while i < length:
            if start+i < self.Tperf:
                noiselessTS[:, i] = \
                    self.generateNoiselessFrom_A(
                        4 + 1,
                        self.P_all[:, self.Tstart + start + i - 4: self.Tstart + start + i].copy(),
                        4
                    )[:, -1]
                # noiselessTS[:, i] = self.generateNoiselessFromA(2, self.Pperf[:, start+i])[:, 1]
                i += 1
            else:
                i += 1
        return noiselessTS


    # s(t+1)=A*s(t)+sigma*Z(t)
    # Estimate the matrix A above from a given time series s
    # using Moore-Penrose pseudoinverse
    # @author R. Sipos
    def est_A_pinv(self):
        dim = self.Pid.shape
        Tw = dim[1] - 1
        # below works on s'
        A = np.zeros((dim[0], dim[0]))

        An_coeff = np.zeros((dim[0], dim[0]))
        An_coeff = self.Pid[dim[0] - Tw:dim[0] - 1, 1:dim[0]] * self.Pid[dim[0] - Tw:dim[0] - 1, 1:dim[0]]

        An_inv = np.zeros((dim[0], dim[0]))  # ok<NASGU>
        if np.linalg.matrix_rank(An_coeff) == dim:
            An_inv = np.invert(An_coeff)
        else:
            An_inv = np.linalg.pinv(An_coeff)  # Moore-Penrose pseudoinverse

        for n in range(dim):
            b = np.zeros(dim[0], 1)
            b[range(dim[0])] = self.Tid[dim[1] - Tw + 1:dim[1], n] * self.Tid[dim[1] - Tw:dim[1] - 1, 1:dim[0]]
            A[n, :] = An_inv * b

        print('done')

        return A

    def makeZeroMean(self):
        self.mean = np.mean(self.Pid, axis=1)
        self.P_all = (self.P_all.transpose() - self.mean).transpose()
        self.Pid = (self.Pid.transpose() - self.mean).transpose()
        self.Pperf = (self.Pperf.transpose() - self.mean).transpose()

    def meanEstimationError(self):
        self.reconstructOriginal()
        if self.useShift:
            return np.linalg.norm(self.recPperf[:, :min(30, len(self.tr_Pperf))]-self.tr_Pperf[:, :min(30, len(self.tr_Pperf))])
        else:
            return np.linalg.norm(self.recPperf[:, :min(30, len(self.tr_Pperf))] - self.tr_Pperf[:, :min(30, len(self.tr_Pperf))])

    def estAVarP(self, p):
        size = self.Pid.shape[0]
        varModel = VAR(self.Pid.transpose())
        self.result = varModel.fit(p)
        stackedPid = self.Pid.copy()
        if p > 1:
            for i in range(p-1):
                stackedPid = np.vstack((stackedPid, np.roll(self.Pid, -(i+1))))
        cov = np.cov(stackedPid)
        covs = np.full((p, p, size, size), 0.0)
        for i in range(p):
            for j in range(p):
                covs[i, j, :, :] = cov[i * size:(i + 1) * size, j * size:(j + 1) * size]
        self.Ap = np.zeros((size, size))
        for i in range(p):
            for j in range(p):
                self.Ap += la.calcATGB(self.result.coefs[j], covs[i, j, :, :], self.result.coefs[i])
        self.A = self.Ap.copy()
        return self.Ap

    def estAmatrixVarP(self, p, matrixIdx = 0):
        size = self.Pid.shape[0]
        varModel = VAR(self.Pid.transpose())
        self.result = varModel.fit(p)
        return self.result.coefs[matrixIdx]

    def estA(self):
        # s(t+1)=A*s(t)+sigma*Z(t)
        # Estimate the matrix A above from a given time series s
        # using Moore-Penrose pseudoinverse

        # Add a new constant 1s timeSeries to self.Pid in order to incorporate the shift in the dynamics.
        # s(t+1)=A*s(t)+sigma*Z(t)+shift
        if self.useShift:
            self.Pid = np.vstack((self.Pid, np.ones(self.Pid.shape[1])))
        dim = self.Pid.shape
        # Estimation method for A depends on n>=t vs t<n
        if dim[0] >= dim[1]:
            # underdetermined case for A, we can solve st=A*s(t-1) exactly
            st1 = self.Pid[:, range(dim[1] - 1)]
            st = self.Pid[:, range(1, dim[1])]
            if np.isnan(st).any() | np.isnan(st1).any() | np.isinf(st).any() | np.isinf(st1).any():
                print("estA dim0 > dim1", np.isnan(st).any(), np.isnan(st1).any(), np.isinf(st).any(),
                      np.isinf(st1).any())
            self.A = np.matmul(st, np.linalg.pinv(st1))
            # self.A = np.matmul(st, np.matmul(st1.transpose(), np.linalg.inv(np.matmul(st1, st1.transpose()))))
            self.A[-1, :-1] = 0
            self.A[-1, -1] = 1
        else:
            # Compute A_hat=(s^T_t-1*s_t-1)^(-1)*(s^T_t-1*S_t)
            B = np.zeros((dim[0], dim[0]))
            C = np.zeros((dim[0], dim[0]))
            D = np.zeros((dim[0], dim[0]))
            E = np.zeros((dim[0], dim[0]))
            for i in range(dim[0]):
                for j in range(dim[0]):
                    for k in range(1, dim[1]):
                        B[i, j] += self.Pid[i, k - 1] * self.Pid[j, k - 1]
                        C[i, j] += self.Pid[i, k] * self.Pid[j, k - 1]
            for k in range(1, dim[1]):
                D += np.outer(self.Pid[:,k],self.Pid[:,k-1])
                E += np.outer(self.Pid[:, k-1], self.Pid[:, k - 1])
            # should be pinv(B)*C, not clear why below works...
            # probably i and j above are mixed up somehow.
            self.A = np.matmul(C, np.linalg.pinv(B))

        if self.useShift:
            self.est_cshift = np.mean(self.A[:-1,-1])
            self.calib_cshift = self.A[:-1,-1]
            # Now remove the constant 1s from self.Pid and its effect from A
            self.Pid = self.Pid[:-1, :]
            self.Aext = self.A
            self.A = self.A[:-1, :-1]
        else:
            self.Aext = self.A
            self.est_cshift = 0
            self.calib_cshift = np.zeros(dim[0])

        # return A



    def est_G_lyap_numeric_corrnoise(self):
        self.correlationType = True
        n = self.Pid.shape
        # first, estimate noise covariance
        sigma_sq_hat = self.est_noise_correl()

        # Solving the Lyapunov equation numerically
        # in order to ensure positive definiteness

        # paramters needed for running numerical sol
        delta = 0.1
        maxIter = 20000
        tol = 0.01
        lyapdist = np.zeros((maxIter, 1))
        # start with sample covariance
        G_new = np.cov(self.Pid)
        # ATGA = np.matmul(np.transpose(A), np.matmul(G_new, A))+sigma_sq_hat
        ATGA = la.calcATGA(self.A, G_new) + sigma_sq_hat
        bestlyap = np.linalg.norm(G_new - ATGA)
        G_best = G_new
        for i in range(maxIter):
            G_old = G_new
            G_new = G_old - delta * (G_old - ATGA)

            if not np.all(np.linalg.eigvals(G_new) > 0):
                G_new = corrtools.cov_nearest(G_new)

            # check if G_new is posi defi. This will display error if not.
            # NOTE: chol appears unreliable for large eigenvalues, just means
            # G is divergent.
            try:
                R = np.linalg.cholesky(corrtools.cov_nearest(G_new))
            except:
                print('Divergent G - stopping numerical iteration.')
                break

            # check how we did...
            # ATGA = np.matmul(np.transpose(A), np.matmul(G_new, A))+sigma_sq_hat
            ATGA = la.calcATGA(self.A, G_new) + sigma_sq_hat
            lyapdist[i] = np.linalg.norm(G_new - ATGA)
            # save the best in case we become divergent...
            if (lyapdist[i] < bestlyap):
                G_best = G_new
                bestlyap = lyapdist[i]
            if np.linalg.norm(G_new - G_old) < tol:
                print('Converged to within movement tolerance.')
                break

        # figure;
        # plot (lyapdist);
        self.G = G_best
        # return G_best

    # s(t+1)=A*s(t)+Z(t)
    # Estimate Z(t) given s and A.
    # @Author=N Fogarasi
    def est_noise_correl(self):
        n = self.Pid.shape
        summ = np.zeros((n[0], n[0]))
        for i in range(n[1] - 1):
            d = self.Pid[:, i + 1] - self.A.dot(self.Pid[:, i])
            diff = np.outer(d, d)
            summ += diff
        return summ / n[1]

    def est_G_lyap(self):
        # s(t+1)=A*s(t)+sigma*Z(t)
        # Estimate the correlation matrix G from above from a given time series s
        # and given an estimate for A
        self.correlationType = False
        dim = self.Pid.shape
        # first, estimate sigma
        sigma_sq_hat = self.est_sigma2() * np.eye(dim[0])
        # Solving the Lyapunov equation for G
        from scipy import linalg
        self.G = linalg.solve_discrete_lyapunov(np.transpose(self.A), sigma_sq_hat)
        # self.G = corrtools.cov_nearest(self.G, "nearest")
        # return G_hat


    def est_sigma2(self):
        size = self.Pid.shape
        summ = 0
        diff = np.zeros((size[1] - 1, 1))
        for i in range(size[1] - 1):
            d = self.Pid[:, i + 1] - (self.A.dot(self.Pid[:, i])+self.est_cshift)
            diff[i] = d.dot(d)
            summ += diff[i]

        return summ / size[0] / size[1]  # sigma2

    def selectModel(self, n_diff):
        if self.model == 'regular':
            pass
        elif self.model == 'log':
            self.log()
        elif self.model == 'diff':
            self.differentiate(n_diff)
        elif self.model == 'logdiff':
            self.log()
            self.differentiate(1)
        self.setIdandPerf()

    def differentiate(self, n_diff=1):
        self.P_all = np.diff(self.P_all, n_diff)

    def log(self):
        self.P_all = np.log(self.P_all)

    def tr(self):
        self.tr_P_all = self.P_all
        self.tr_Pid = self.tr_P_all[:, range(self.Tstart-self.Tid, self.Tstart)]
        self.tr_Pperf = self.tr_P_all[:, range(self.Tstart, self.Tperf + self.Tstart)]

    def reconstructOriginal(self):
        if self.model == 'regular':
            if self.useShift:
                self.recPperf = self.generateNoiselessFromA(30, self.Pperf[:,0])[:-1, :] + np.tile(self.mean.transpose(), (30, 1)).transpose()
            else:
                self.recPperf = self.generateNoiselessFromA(30, self.Pperf[:,0]) + np.tile(self.mean.transpose(), (30, 1)).transpose()
        elif self.model == 'diff':
            if self.useShift:
                self.recPperf = np.tile(self.tr_Pperf[:, 0].transpose(), (30, 1)).transpose() + np.tile(self.mean.transpose(), (30, 1)).transpose() + np.cumsum(self.generateNoiselessFromA(30, self.Pperf[:,0]), axis=1)[:-1, :]
            else:
                self.recPperf = np.tile(self.tr_Pperf[:, 0].transpose(),(30,1)).transpose() + np.tile(self.mean.transpose(), (30, 1)).transpose() + np.cumsum(self.generateNoiselessFromA(30, self.Pperf[:,0]), axis=1)
        elif self.model == 'log':
            if self.useShift:
                self.recPperf = np.exp(self.generateNoiselessFromA(30, self.Pperf[:,0])[:-1, :] + np.tile(self.mean.transpose(), (30, 1)).transpose())
            else:
                self.recPperf = np.exp(self.generateNoiselessFromA(30, self.Pperf[:,0]) + np.tile(self.mean.transpose(), (30, 1)).transpose())
        elif self.model == 'logdiff':
            if self.useShift:
                self.recPperf = np.exp(np.log(np.tile(self.tr_Pperf[:, 0].transpose(),(30,1)).transpose()) + np.tile(self.mean.transpose(), (30, 1)).transpose() + np.cumsum(self.generateNoiselessFromA(30, self.Pperf[:,0]), axis=1)[:-1,:])
            else:
                self.recPperf = np.exp(np.log(np.tile(self.tr_Pperf[:, 0].transpose(),(30,1)).transpose()) + np.tile(self.mean.transpose(), (30, 1)).transpose() + np.cumsum(self.generateNoiselessFromA(30, self.Pperf[:,0]), axis=1))

    def setIdandPerf(self):
        self.Pid = self.P_all[:, range(self.Tstart-self.Tid, self.Tstart)]
        self.Pperf = self.P_all[:, range(self.Tstart, self.Tperf + self.Tstart)]

    def resetData(self):
        self.P_all = self.tr_P_all
        self.setIdandPerf()

    def solve_trunc(self):
        # Truncation method: take max L components of the eigenvector
        # and using those dimensions, find max eigenvalue
        # @Author: Norbert Fogarasi
        n = self.A.shape
        # ATGA = np.matmul(np.transpose(A), np.matmul(G, A))  # Fixed NF 20120802
        ATGA = la.calcATGA(self.A, self.G)
        [maxvec, dmax] = la.max_eigvector(ATGA, self.G)
        wsort = np.sort(maxvec)
        ix = la.getPermutationOrder(maxvec, wsort)
        trun_ix = ix[range(n[0] - self.L, n[0])]
        s_trun_ix = np.sort(trun_ix)
        # Find eigenval corresponding to truncated matrix
        if np.setdiff1d(range(n[0]), s_trun_ix).size == 0:
            return maxvec, dmax
        else:
            diffVector = np.setdiff1d(range(n[0]), s_trun_ix)
            Gcurr = np.delete(self.G, diffVector, 0)
            Gcurr = np.delete(Gcurr, diffVector, 1)
            # Gcurr = G[s_trun_ix, s_trun_ix]
            ATGAcurr = np.delete(ATGA, diffVector, 0)
            ATGAcurr = np.delete(ATGAcurr, diffVector, 1)
            # ATGAcurr = ATGA[s_trun_ix, s_trun_ix]
            [xopt, lamb] = la.max_eigvector(ATGAcurr, Gcurr)
            x = np.zeros(n[0])
            x[s_trun_ix] = xopt
            return x, lamb

    def exhaust_calculation(self, i):
        n = self.A.shape[0]
        ATGA = self.ATGA
        diffVector = np.setdiff1d(range(n), i)
        ATGAcurr = np.delete(ATGA, diffVector, 0)
        ATGAcurr = np.delete(ATGAcurr, diffVector, 1)
        Gcurr = np.delete(self.G, diffVector, 0)
        Gcurr = np.delete(Gcurr, diffVector, 1)
        [Vcurr, Dcurr] = la.max_eigvector(ATGAcurr, Gcurr)
        return Vcurr, Dcurr, i

    def solve_exhaust(self):
        # Exhaustive search to find solution of generalized eigenvalue problem
        # under cardinality constraint
        n = self.A.shape
        from itertools import combinations
        C = list(combinations(range(n[0]), self.L))
        numeig = len(C)
        maxeigval = 0
        maxeigvec = np.ones(self.L)
        eigpos = 1
        currval = np.zeros(numeig)
        # ATGA=A'*G*A;
        self.ATGA = la.calcATGA(self.A, self.G)  #### FIXED NF 20120802
        ATGA = self.ATGA
        # Dcurr_ = []
        # Vcurr_ = []
        # Ccurr_ = []

        # import time
        # import concurrent.futures
        #
        # start = time.perf_counter()
        #
        # with concurrent.futures.ProcessPoolExecutor(max_workers=13) as executor:
        #     for res in executor.map(self.exhaust_calculation, C):
        #         Vcurr_.append(res[0])
        #         Dcurr_.append(res[1])
        #         Ccurr_.append(res[2])
        #     # currval = executor.map(self.exhaust_calculation, ATGA, n, C)0
        #     # currval = [executor.submit(self.exhaust_calculation, ATGA, n, c) for c in C]
        #     finish = time.perf_counter()
        #     print(f'Finished in {finish - start} seconds.')
        #
        #     # for future in concurrent.futures.as_completed(currval):
        #     #     Vcurr_.append(future.result()[0])
        #     #     Dcurr_.append(future.result()[1])
        #     #     Ccurr_.append(future.result()[2])
        #
        # maxeigval_ = max(Dcurr_)
        # maxeigvec_ = Vcurr_[Dcurr_.index(max(Dcurr_))]
        #
        # x = np.zeros(n[0])
        # np.put(x, Ccurr_[Dcurr_.index(max(Dcurr_))], maxeigvec_)


        j = 0
        for i in C:
            diffVector = np.setdiff1d(range(n[0]), i)
            ATGAcurr = np.delete(ATGA, diffVector, 0)
            ATGAcurr = np.delete(ATGAcurr, diffVector, 1)
            Gcurr = np.delete(self.G, diffVector, 0)
            Gcurr = np.delete(Gcurr, diffVector, 1)
            # [Vcurr,Dcurr]=eigs(ATGAcurr,Gcurr,1);
            [Vcurr, Dcurr] = la.max_eigvector(ATGAcurr, Gcurr)
            # ecord curr eigenvalue
            currval[j] = Dcurr

            if currval[j] > maxeigval:
                maxeigval = currval[j]
                maxeigvec = Vcurr
                eigpos = j

            j += 1

        # reconstruct full w vector
        x = np.zeros(n[0])
        np.put(x, C[eigpos], maxeigvec)

        # finish2 = time.perf_counter()
        #
        # print(f'Finished in {finish2 - finish} seconds.')
        # lamb = la.max_eigvector(ATGA, self.G)
        # lamb = np.matmul(ATGA, x)
        # lamb = np.matmul(x, lamb)
        # lamb /= np.matmul(x, np.matmul(self.G, x))
        # return x, lamb
        return x, maxeigval, currval, C

    def solve_greedy(self):
        # Greedy algorithm to solve the generalized eigenvalue problem
        # under cardinatlity constraint
        dim = self.A.shape
        # fullATGA = np.matmul(np.transpose(A), np.matmul(G, A))
        # fullATGA = np.matmul(A, np.matmul(G, np.transpose(A))) # FIXED NF 20120802
        # fullATGA = la.calcATGA(self.A, self.G)
        fullATGA = self.estAVarP(self.VARparameter)
        fullG = self.G
        traceLambda = np.zeros(self.L)
        if self.L == dim[0]:
            [x, lamb] = la.max_eigvector(fullATGA, self.G)
            return x, lamb, lamb * np.ones(self.L)
        else:
            volmax = -np.inf
            xopt = np.zeros((self.L, 1))
            I = np.array([], dtype=int)
            k = 0
            for k in range(self.L):
                lastidx = -1
                for i in np.setdiff1d(range(dim[0]), I):
                    Ii = np.insert(I, I.size, i)
                    if Ii.size > 1:
                        diffVector = np.setdiff1d(range(dim[0]), Ii)
                        Gcurr = np.delete(fullG, diffVector, 0)
                        Gcurr = np.delete(Gcurr, diffVector, 1)
                        ATGAcurr = np.delete(fullATGA, diffVector, 0)
                        ATGAcurr = np.delete(ATGAcurr, diffVector, 1)
                        [x, vol] = la.max_eigvector(ATGAcurr, Gcurr)
                    else:
                        ATGA = fullATGA[Ii, Ii]
                        G = fullG[Ii, Ii]
                        try:
                            vol = ATGA / G
                        except:
                            print(G)
                        x = 1
                    if vol > volmax:
                        volmax = vol
                        xopt = x
                        lastidx = i
                        traceLambda[k] = volmax
                if lastidx == -1:
                    break
                else:
                    I = np.insert(I, I.size, lastidx)  # ok<AGROW>
            # Revert to original space.
            fullx = np.zeros(dim[0])
            if sum(I.shape) != 0:
                fullx[np.sort(I)] = xopt
            x = fullx  # disp('Vector x identified.');
            lamb = volmax
            return x, lamb, traceLambda




    def parameterEstimation(self):
        # Pid = P_all[:, range(Tid)]
        # Pperf = P_all[:, range(Tid, Tid + Tperf)]

        #######################################
        # Step 1. Parameter Estimation
        #######################################
        # self.estA()
        self.estAVarP(self.VARparameter)
        sigma2 = self.est_sigma2()
        # display error just for debugging
        sigerr = sigma2 - self.sigma * self.sigma
        if self.simulated:
            Aerr = np.linalg.norm(self.A_true - self.A)

        # estimate G
        self.G = np.cov(self.Pid)
        # self.est_G_lyap()
        # try:
        #     np.linalg.cholesky(self.G)
        # except:
        #     # print('Exact Lyapunov solution is non posi defi. Using numerical solution.')
        #     self.G = corrtools.cov_nearest(self.G, "nearest")
        #     self.est_G_lyap_numeric_corrnoise()
        # G2 = np.cov(Pid)
        self.Beta = np.linalg.norm(self.G - np.cov(self.Pid))
        # return A, G, Beta

def valami(weight, A, G):
    n = A.shape
    ATGA = la.calcATGA(A, G)
    diffVector = np.setdiff1d(range(n[0]), np.nonzero(weight))
    ATGAcurr = np.delete(ATGA, diffVector, 0)
    ATGAcurr = np.delete(ATGAcurr, diffVector, 1)
    Gcurr = np.delete(G, diffVector, 0)
    Gcurr = np.delete(Gcurr, diffVector, 1)
    # [Vcurr,Dcurr]=eigs(ATGAcurr,Gcurr,1);
    [Vcurr, Dcurr] = la.max_eigvector(ATGAcurr, Gcurr)
    # ecord curr eigenvalue
    return Vcurr, -Dcurr


def canonicalComm(A, G):
    return np.dot(A, G) - np.dot(G, A)