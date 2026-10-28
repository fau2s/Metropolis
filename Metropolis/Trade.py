import numpy as np
import sys
# from matplotlib import pyplot
from matplotlib import pyplot

sys.path.append("annealing.py")
from Metropolis import estimateA as estA
from Metropolis import estimateG as estG
from Metropolis import PortfolioSelection as ps
from Metropolis import estimateSigma2 as estS2
from Metropolis import estimateMu as estMu
from Metropolis import GenerateVAR1 as gen
from Metropolis import SparsePortfolioSelection as sps
from Metropolis import Trading as trd
from Metropolis import evaluation as eval
from Metropolis import parameterEstimation as pe
# Define input parameters for the run

histplot = 1

loadfile = 0  # whether to run from a file or generate data
repeats = 1  # number of repeats in case of generated data
methods = [2, 3, 4]  # exhaustive,greedy,SA,trunc

n = 13   # Total number of assets to choose from
L = 10# Max number of assets allowed in sparse portfolio
Tid = 160   # Size of identification window
Tperf = 100   # Size of trading window
C0 = 100  # Amount of cash available for trading
r = 0  # daily compounded risk free interest rate

# Define internal model parameters
sigma = 0.01  # size of the noise in generated process
tw = 50  # size of time window used for estimating G
eps1 = 0.98  # below what ratio of mu we buy
eps2 = 1.02  # above what ratio of mu we sell to take profit
risktaker = 1  # wager all our money at each opportunity
recalibrate = True
paralelPortfolio = False

#######################################
# Step 0. Generate or Load data
#######################################
lam = np.zeros((repeats, 4 ))
runtime = np.zeros((repeats, 4))
profit = np.zeros((repeats, 4))
numTrade = np.zeros((repeats,4))

Beta = np.zeros(repeats)
Return = np.zeros((repeats, 4))

# pyplot.show()

for rep in range(repeats):

    print('In the cycle #', rep)

    #######################################
    # Step 0. Gathering data
    #######################################

    if loadfile == 1:
        P_all = np.loadtxt('..\\Norbert\\26fxspot.csv')
        import scipy.io as sio
        sio.loadmat('..\\Norbert\\P_20110622')
        # load('P_20110622')
        # load ('P_usdlibor-all') 
        # load ('s_0517_271')  P_all=s(:,2:271)  # off by one bogosity
        
    else:
        maxlam = 0
        while maxlam < 0.05:  # set limit here
            [A_true, P_all, cshift] = gen.generateVAR1(n, Tid+Tperf, sigma)
            Pid = P_all[:, range(Tid)]
            A = estA.estA(Pid)
            G = estG.est_G_lyap_numeric_corrnoise(Pid, A)
            [x, maxlam] = ps.solve_trunc(A, G, n)
            # print(maxlam)

    tact = Tid + 1

    while tact < Tid + Tperf:
        Pid = P_all[:, range(Tid)]
        Pperf = P_all[:, range(Tid, Tid + Tperf)]

        if paralelPortfolio & recalibrate:
            pass
    #######################################
    # Step 1. Parameter Estimation
    #######################################
            [A, G, Beta[rep]] = pe.parameterEstimation(Pid, Pperf, A_true, sigma, Beta)
    # A = estA.estA(Pid)
    # sigma2 = estS2.est_sigma2(Pid, A)
    # # display error just for debugging
    # sigerr = sigma2 - sigma * sigma
    # Aerr = np.linalg.norm(A_true - A)
    #
    # # estimate G
    # G = estG.est_G_lyap(Pid, A)
    # try:
    #     R = np.linalg.cholesky(G)
    # except:
    #     # print('Exact Lyapunov solution is non posi defi. Using numerical solution.')
    #     G = estG.est_G_lyap_numeric_corrnoise(Pid, A)
    # G2 = np.cov(Pid)
    # Beta[rep] = np.linalg.norm(G - G2)

    #######################################
    # Step 2. Sparse Portfolio Selection
    #######################################

            for meth in methods:  # methods: exhaustive, greedy, SA, trunc

                [xopt, lamb, xoptbuy, xoptnorm, mu0] = sps.sparsePortfolioSelect(A, G, L, meth, C0, Pid, Tid)

        # [xopt, lamb] = ps.sparse_meanrev_portfselect(A, G, L, meth)
        # # runtime[rep,meth]=toc
        # lam[rep, meth-1] = lamb
        # if meth == 1:
        #     wexh = xopt  # save wexh for annealing.py comparison
        #
        # # Allow shorting, but display a warning message
        # if np.sum(xopt >= 0) != n:
        #     print('Negative element in xopt - will need to do shorting')
        #
        # # Scale portfolio so that we  sp   exactly C0
        # xoptnorm = xopt / sum(abs(xopt))  # normalize xopt so that sum = 1
        # xoptsp = C0 * xoptnorm  # amount of money to sp   on each asset
        # xoptbuy = np.floor(xoptsp / Pid[:, Tid-1])  # number of contracts to buy of each asset
        # xoptbuy0 = xoptbuy
        # # Compute historical value of xoptbuy portfolio during Tid
        # histportfval = np.matmul(xoptbuy, Pid)
        #
        # # Estimate the mu associated with xoptbuy
        # mu0 = estMu.est_mu_linreg(histportfval)
        #######################################
        # Step 3. Trading
        #######################################
                [portfheld, V, C, numtrade] = trd.Trading(Tperf, eps1, eps2, mu0, xoptbuy, xoptnorm, Pid, P_all, Pperf, Tid, C0, risktaker)

            tact += 1
        # portfheld = False
        # buyt = np.array([], dtype=int)
        # buyp = np.array([], dtype=int)
        # sellt = np.array([], dtype=int)
        # sellp = np.array([], dtype=int)
        # mu = np.zeros(Tperf+1)
        # C = np.zeros(Tperf+1)  # amount of cash held at time i
        # x = np.zeros((Tperf+1, n))  # number of assets held
        # V = np.zeros(Tperf+1)  # value of assets held at time i
        # numtrade = 0  # number of trades performed
        # # Check whether to trade immediately
        # if eps1*mu0 > xoptbuy.dot(Pid[:, Tid-1]):
        #     V0 = xoptbuy.dot(Pid[:, Tid-1])
        #     # print('0. BUY at price ', str(V0))
        #     portfheld = True
        #     buyt = np.insert(buyt, buyt.size, Tid)
        #     buyp = np.insert(buyp, buyp.size, V0)
        #     C[0] = C0-V0
        #     x[0, :] = xoptbuy
        #     numtrade += 1
        # else:
        #     C[0] = C0
        #     x[0, :] = np.zeros(n)
        #     V0 = 0
        #
        # for i in range(Tperf):
        #     # compute mu in terms of orig portfolio
        #     mu[i] = estMu.est_mu_linreg(xoptbuy0.dot(P_all[:, range(Tid + i)]))
        #     currval = xoptbuy0.dot(Pperf[:, i])
        #     if not portfheld:
        #         # check if we should buy
        #         if eps1*mu[i] > currval:
        #             portfheld = True
        #             if risktaker == 1:
        #                 # Case 1: wager all our money
        #                 xoptsp = C[i]*xoptnorm
        #
        #             # Case 2: we only sp   C0, so do not recompute xoptsp
        #             xoptbuy = np.floor(xoptsp / Pperf[:, i])
        #             V[i] = xoptbuy.dot(Pperf[:, i])
        #             C[i] = C[i]-V[i]
        #             C[i+1] = C[i]
        #             x[i+1, :] = xoptbuy
        #             numtrade += 1
        #             # print([str(i), '. BUY at price ', str(V[i])])
        #             buyt = np.insert(buyt, buyt.size, Tid+i)
        #             buyp = np.insert(buyp, buyp.size, currval)
        #         else:
        #             C[i+1] = C[i]
        #             V[i] = 0
        #             x[i+1, :] = x[i, :]
        #
        #     else:
        #         # compute portf value at new price
        #         V[i] = x[i, :].dot(Pperf[:, i])
        #         # check if we should sell
        #         if eps2*mu[i] < currval:
        #             portfheld = False
        #             C[i+1] = C[i]+x[i, :].dot(Pperf[:, i])
        #             V[i+1] = 0
        #             x[i+1, :] = np.zeros(n)
        #             numtrade += 1
        #             # print([str(i), '. SELL at price ', str(V[i])])
        #             sellt = np.insert(sellt, sellt.size, Tid+i)
        #             sellp = np.insert(sellp, sellp.size, currval)
        #         else:
        #             # hold.
        #             C[i+1] = C[i]
        #             x[i+1, :] = x[i, :]

        #######################################
        # Step 4. Close off and evaluation
        #######################################
            [Return[rep, meth-1], profit[rep, meth-1], numTrade[rep, meth-1]] = eval.evaluation(V, Pperf, Tperf, x, portfheld, C, C0, numtrade)
        # Cfinal = 0
        # V[Tperf] = 0
        # if portfheld:
        #     # print('CASH IN at the   .')
        #     Cfinal = C[Tperf-1] + x[Tperf, :].dot(Pperf[:, Tperf-1])
        # else:
        #      Cfinal = C[Tperf]
        #
        # Return[rep, meth-1] = 100*(Cfinal-C0)/C0
        # # print(['Final cash is ', str(Cfinal)])
        # # print(['A return of ', str(Return), ' #'])
        # # print(['We made ', str(numtrade), ' trades.'])
        # profit[rep, meth-1] = Cfinal-C0
        # numTrade[rep, meth-1] = numtrade
        # pyplot.plot(range(Tperf+1), C + V, range(Tperf+1), C)
        # pyplot.gcf().clear()
        # pyplot.hist(Return,30)
        # figure
        # Plot orig portfolio value and mu estimate
        # plot (10:Tid+Tperf,xoptbuy0'*P_all(:,10:Tid+Tperf),Tid+1:Tid+Tperf,mu)
        # Plot buy and sell pointspyplot.plot(range(Tperf), C + V, range(Tperf+1), C)
        # hold on
        # plot (buyt,buyp,'rs',sellt,sellp,'rh')
        # hold off
        # leg  ('Portfolio Value','Mean Estimate','Buy Action',...
        #     'Sell Action', 'Location', 'SouthWe
        #
np.savez('.\\data_n13_L10_Tid160_Tperf100',lam, profit, numTrade, Beta)
pyplot.plot(range(Tperf + 1), C + V, range(Tperf + 1), C)

# plot histogramic results
 
# if (histplot==1):
#     x=[-200:100:4000]
#     bar(x,histc(profit(:,1),x))
#     title ('Exhaustive Method Trading Results')
#     xlabel ('Profit generated (C0=100)')
#     ylabel ('Frequency (out of 2000 cases)')
#     figure
#     bar(x,histc(profit(:,2),x))
#     title ('Greedy Method Trading Results')
#     xlabel ('Profit generated (C0=100)')
#     ylabel ('Frequency (out of 2000 cases)')
#     figure
#     bar(x,histc(profit(:,3),x))
#     title ('Simulated Annealing Trading Results')
#     xlabel ('Profit generated (C0=100)')
#     ylabel ('Frequency (out of 2000 cases)')
#     figure
#     bar(x,histc(profit(:,4),x))
#     title ('Truncation Method Trading Results')
#     xlabel ('Profit generated (C0=100)')
#     ylabel ('Frequency (out of 2000 cases)')
   


