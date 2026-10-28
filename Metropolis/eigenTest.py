from Metropolis import data
import numpy as np
import pandas as pd
from Metropolis import linAlg as la
from matplotlib import pyplot as pl
import xlrd

isSimulated = False

start = 3
n = 50

fullTickerList = pd.read_excel('E:\PhD_\Yahoo Ticker Symbols - September 2017.xlsx', 'Stock', header=3, usecols=3)

tickerList = fullTickerList['Ticker'][:200]

# tickerList = ['AAPL', 'MSFT', 'INTC', 'WFC', 'BAC', 'C', 'GATA', 'COLM', 'ACB', 'CLSD', 'NOW', 'TXN', 'FNMA',
#               'PYPL', 'F', 'CCEP', 'PG', 'SBUX', 'JNJ', 'ABBV', 'PEGA', 'SQM', 'NKTR', 'NBIX', 'ELCPF', 'EXAS',
#               'AMZN', 'GOOG', 'KOF', 'CHA', 'LFC', 'CTRN', 'ON', 'KEP', 'CHL', 'SBUX', 'NVDA', 'CSCO', 'VOD',
#               'MRVL', 'GPRO', 'GPRE', 'GRPN', 'HCSG', 'GLUU', 'JCOM', 'GOLD', 'MYGN', 'RGLD', 'EA', 'TSLA',
#               'FDX', 'FB', 'TWTR', 'SHOP', 'AVGO', 'GE', 'AVP', 'DHR', 'MU', 'CRARF', 'MLHR', 'LII', 'EHTH',
#               'HEI', 'SRPT', 'TTC', 'NTRA', 'JBL', 'TXG', 'UPS', 'PATK', 'NEO', 'DDD', 'CVCO', 'IBKC',
#               'SPHS', 'BBQ', 'FNLC', 'NVCR', 'POPE', 'AGLE', 'SLAB', 'FEYE', 'FARO', 'FARM', 'SCOR',
#               'BBI', 'NKE', 'WMT', 'DIS', 'BA', 'TRV', 'KO', 'MMM', 'AXP', 'MCD', 'V', 'GS', 'XOM', 'WBA',
#               'HD', 'CVX', 'JPM', 'SHW', 'APD', 'MT', 'SCCO', 'XRX', 'WFC']
# tickerList = ['SCCO', 'EURUSD=X', 'GC=F' ]

print(len(tickerList))

start = 3
n = 30


if isSimulated:

    numberOfData = int((n*(n+1)/2-start*(start-1)/2)*500)

    df = pd.DataFrame({'sigma': np.ones(numberOfData),
                        'size': np.ones(numberOfData),
                        'sparsity': np.ones(numberOfData),
                        'noise std est': np.ones(numberOfData),
                        'shift': np.ones(numberOfData),
                        'shift est': np.ones(numberOfData),
                        'norm': np.ones(numberOfData),
                       'norm_2': np.ones(numberOfData),
                        'lambda_ex': np.ones(numberOfData),
                        'lambda_gr': np.ones(numberOfData),
                        'lambda_ex_Atrue': np.ones(numberOfData),
                        'lambda_ex_2': np.ones(numberOfData),
                        'lambda_gr_2': np.ones(numberOfData)})

    df2 = pd.DataFrame({'sigma': np.ones(15*(n-start+1)),
                        'size': np.ones(15*(n-start+1)),
                        'noise std est': np.ones(15*(n-start+1)),
                        'shift': np.ones(15*(n-start+1)),
                        'shift est': np.ones(15*(n-start+1)),
                        'norm': np.ones(15*(n-start+1))})
                        # 'norm': np.ones((15*(n-start+1), n))})



    dfShifts = np.ones((15*(n-start+1),n))


    diffProb = np.zeros([(n-start+1), 1])
    meanDiffEx = np.zeros([1, (n-start+1)])
    meanDiffGr = np.zeros([1, (n-start+1)])

numberOfData = int((n*(n+1)/2-start*(start-1)/2)*500)

df = pd.DataFrame({'sigma': np.ones(numberOfData),
                    'size': np.ones(numberOfData),
                    'sparsity': np.ones(numberOfData),
                    'noise std est': np.ones(numberOfData),
                    'shift': np.ones(numberOfData),
                    'shift est': np.ones(numberOfData),
                    'norm': np.ones(numberOfData),
                   'norm_2': np.ones(numberOfData),
                    'lambda_ex': np.ones(numberOfData),
                    'lambda_gr': np.ones(numberOfData),
                    'lambda_ex_Atrue': np.ones(numberOfData),
                    'lambda_ex_2': np.ones(numberOfData),
                    'lambda_gr_2': np.ones(numberOfData)})

diffProb = np.zeros([n-start+1, 1])
meanDiffEx = np.zeros([1, n-start+1])
meanDiffGr = np.zeros([1, n-start+1])

k = 0


cshifts = np.zeros((15*(n-start+1), len(tickerList)))

tickerShiftDF = pd.DataFrame(cshifts, columns=tickerList)

# for i in range(start, n, 1):
#
#     for sigma in np.arange(0.1, 6, 0.4):
#         print('Construting...')
#         tickers = np.random.choice(tickerList, i, replace=False)
#         testData = data.Data(i, i, sigma, 300, False, True, tickers)
#         # testData2 = data.Data(i, i, sigma, 300, True, True)
#         df2['size'][k] = i
#         df2['sigma'][k] = sigma
#         df2['noise std est'][k] = testData.est_sigma2()
#         df2['shift'][k] = testData.cshift
#         print('Estimating...')
#         testData.parameterEstimation()
#         # testData2.parameterEstimation()
#         print('Generating...')
#         testApple = testData.generateNoiselessFromA(30)
#         testData.useShift = False
#         testData.parameterEstimation()
#         testApple2 = testData.generateNoiselessFromA(30)
#         # for t, tt in enumerate(tickers):
#         # # for t,tt in enumerate(testData.Pperf):
#         #     pl.clf()
#         #     pl.xlabel('Time [Days]')
#         #     pl.ylabel('Asset value [$]')
#         #     pl.title('Value of simulated assets vs. estimated noiseless value')
#         #     pl.plot(testApple[t, :30].transpose())
#         #     pl.plot(testApple2[t, :30].transpose())
#         #     pl.plot(testData.Pperf[t, :30].transpose())
#         #     pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift', 'Simulated Asset', ])
#         #     pl.savefig('..\\pictures\\Simulated_estimated_N{i}_S{s}_No{t}.png'.format(i=i, s=sigma, t=t))
#         # df2['shift est'][k] = testData.calib_cshift
#         dfShifts[k,:i] = testData.calib_cshift
#         # for j, tick in enumerate(testData.ticker):
#         #     tickerShiftDF.loc[k, tick] = testData.calib_cshift[j]
#         # df2['norm'][k] = testData.ticker
#         k += 1
#         print(k)
#         # pl.clf()
#         # pl.matshow(testData.A)
#         # pl.colorbar()
#         # pl.savefig('..\\pictures\\Simulated_calibrated_A_N{i}xN{i}_S{s}.png'.format(i=i, s=sigma))
#         # pl.close()
#         # pl.matshow(testData.G)
#         # pl.colorbar()
#         # pl.savefig('..\\pictures\\Simulated_calibrated_G_N{i}xN{i}_S{s}.png'.format(i=i, s=sigma))
#         # np.savetxt('tickers_{ms}.csv'.format(ms=i), testData.ticker, delimiter=',', fmt='%s')
# df2.to_csv('lambda_test.csv')
# np.savetxt('constant_shifts.csv', dfShifts, delimiter=',')
# tickerShiftDF.to_csv('shifts2Tickers.csv')
# exit(0)
sigma = 2
traceLambda = np.ones((500 * start, n))
for sigma in np.arange(0.1, 6, 0.4):
# for iter in np.arange(1, 501):
    for i in range(start, n, 1):
        print('The size of the matrix is {:d}.'.format(i))
        weights = np.ones((5*i, i))
        testData = data.Data(i,i,sigma)
        for j in reversed(range(1, i+1)):
            if j == i:
                kmax = k
            testData.L = j
            df['size'][k] = i
            df['sparsity'][k] = j
            df['sigma'][k] = sigma
            df['noise std est'][k] = testData.est_sigma2()
            df['shift'][k] = testData.cshift
            # df['norm'][k] = np.linalg.norm(testData.A_true - testData.A)
            testData.parameterEstimation()
            testApple = testData.generateNoiselessFromA(200)
            # testData.est_G_lyap_numeric_corrnoise()
            a = testData.solve_exhaust()
            weights[(j-1),:] = a[0]
            df['lambda_ex'][k] = a[1]
            if j == i:
                lexmax = df['lambda_ex'][kmax]
            df['lambda_ex'][k] /= lexmax
            # pl.clf()
            # pl.xlabel('Bins')
            # pl.ylabel('Histogram values')
            # pl.title('Histogram of maximal eigenvalues {iter}'.format(iter=j))
            # pl.hist(a[2] / lexmax, 20)
            # pl.plot((range(i+1,1,-1)), meanDiffEx.transpose()/iter, 'x')

            # if k>0:
            #     if (df['lambda_ex'][k] == df['lambda_ex'][k-1]):
            #         print('Equal lambda!')
            a = testData.solve_greedy()
            weights[(j-1)+i,:] = a[0]
            df['lambda_gr'][k] = a[1]
            # if k == 188:
            #     print('pina')
            if j == i:
                lgrmax = df['lambda_gr'][kmax]
            df['lambda_gr'][k] /= np.absolute(lgrmax)
            traceLambda[k, :j] = a[2] / np.absolute(lgrmax)

            # pl.axvline(df['lambda_gr'][k])
            # pl.axis([1, i, 0, 1.1])
            # pl.savefig('..\pictures\MaxEigenvalue_histogram_N{sz}_S{s}_Iteration{iter}'.format(sz=i, iter=iter, s=j))
            # if k>0:
            #     if (df['lambda_gr'][k] == df['lambda_gr'][k-1]):
            #         print('Equal lambda!')
            #         testData.solve_greedy()
            testData.A = testData.A_true
            a = testData.solve_exhaust()
            weights[(j-1)+2*i,:] = a[0]
            df['lambda_ex_Atrue'][k] = a[1]
            if j == i:
                lextmax = df['lambda_ex_Atrue'][kmax]
            df['lambda_ex_Atrue'][k] /= lextmax
            # if k>0:
            #     if (df['lambda_ex_Atrue'][k] == df['lambda_ex_Atrue'][k-1]):
            #         print('Equal lambda!')
            df['shift est'][k] = testData.est_cshift

            testData.Pid = testData.P_all[:, range(150, testData.Tid + 150)]
            testData.parameterEstimation()
            df['norm_2'][k] = np.linalg.norm(testData.A_true - testData.A)
            a = testData.solve_exhaust()
            weights[(j-1)+3*i,:] = a[0]
            df['lambda_ex_2'][k] = a[1]
            if j == i:
                lex2max = df['lambda_ex_2'][kmax]
            df['lambda_ex_2'][k] /= lex2max
            # if k>0:
            #     if (df['lambda_ex_2'][k] == df['lambda_ex_2'][k-1]):
            #         print('Equal lambda!')
            a = testData.solve_greedy()
            weights[(j-1) + 4*i, :] = a[0]
            df['lambda_gr_2'][k] = a[1]
            if j == i:
                lgr2max = df['lambda_gr_2'][kmax]
            df['lambda_gr_2'][k] /= np.absolute(lgr2max)
            # if k>0:
            #     if (df['lambda_gr_2'][k] == df['lambda_gr_2'][k-1]):
            #         print('Equal lambda!')
            #         testData.solve_greedy()
            k += 1
            testData.Pid = testData.P_all[:, range(0, testData.Tid)]
            # testData.estA()
            testData.parameterEstimation()
            print(k)
            diffProb[j-1] = diffProb[j-1] if abs(df['lambda_ex'][k-1] - df['lambda_gr'][k-1]) < 0.00001 else diffProb[j-1]+1

            # if df['lambda_ex'][k-1] != df['lambda_gr'][k-1] and j==1:
            #     print('pina')
        # if k == 190:
        #     print('pina')
        # meanDiffEx += df['lambda_ex'][k-i:k].values
        # meanDiffGr += df['lambda_ex'][k-i:k].values-np.absolute(df['lambda_gr'][k - i:k].values)
        # f, (ax1, ax2) = pl.subplots(2,1)
        # ax1.set_xlabel('Sparsity')
        # ax2.set_xlabel('Sparsity')
        # ax1.set_ylabel('Eigenvalue')
        # ax2.set_ylabel('Gradient')
        # ax1.plot((range(i, 0, -1)), df['lambda_ex'][k-i:k], 'x')
        # ax1.plot((range(i, 0, -1)), np.absolute(df['lambda_gr'][k-i:k]), 'x')
        # ax2.plot((range(i, 0, -1)), np.gradient(df['lambda_ex'][k-i:k]), 'x')
        # ax2.plot((range(i, 0, -1)), np.gradient(np.absolute(df['lambda_gr'][k-i:k])), 'x')
        # ax1.legend(('Exhaustive','Greedy'))
        # ax2.legend(('Exhaustive', 'Greedy'))
        # ax1.axis([1, i, 0, 1.1])
        # pl.savefig('..\pictures\greedy_exhaustive_gradients_N{sz}_Iteration{iter}'.format(sz=i, iter=iter))
        # pl.clf()
        # pd.DataFrame(weights).to_csv("weights_{sz}.csv".format(sz=i),mode='a')
        # pd.DataFrame(testData.G).to_csv("matrices_{sz}.csv".format(sz=i), mode='a')
        # pd.DataFrame(np.ones(1)*testData.correlationType).to_csv("matrices_{sz}.csv".format(sz=i), mode='a')
        # pd.DataFrame(testData.A).to_csv("matrices_{sz}.csv".format(sz=i), mode='a')
        # pd.DataFrame(testData.A_true).to_csv("matrices_{sz}.csv".format(sz=i), mode='a')
        # pd.DataFrame(la.calcATGA(testData.A, testData.G)).to_csv("matrices_{sz}.csv".format(sz=i), mode='a')
# pl.xlabel('Sparsity')
# pl.ylabel('Mean eigenvalue')
# pl.title('Mean of Exhaustive in terms of sparsity out of {iter}'.format(iter=iter))
# # pl.plot((range(i+1,1,-1)), meanDiffEx.transpose()/iter, 'x')
# pl.plot((range(i+1,1,-1)), meanDiffGr.transpose()/iter, 'x')
# # pl.axis([1, i, 0, 1.1])
# pl.savefig('..\pictures\mean_exhaustive_N{sz}_Iteration{iter}'.format(sz=i, iter=iter))

pl.clf()
pl.plot(testData.Pperf[9,:])
pl.plot(testApple[9:])
pl.plot(testData.Pperf[8,:])
pl.plot(testApple[8,:])
pl.xlabel('Time [Days]')
pl.ylabel('Asset values [$]')
pl.title('Real and VAR(1) modeled asset values')
pl.legend( ['Real asset value 1', 'Approximated value 1', 'Real asset value 2',  'Approximated value 2'])
# pl.savefig('D:\PhD\Real_VAR1_asset_compare_2')
# df.to_csv('lambda_test.csv')
#
# MS=10
# pyplot.plot(shifts[df['sigma']==0.1]['shift'],shifts[df['sigma']==0.1]['shift est'], 'x', markersize=MS)
# pyplot.plot(shifts[df['sigma']==0.3]['shift'],shifts[df['sigma']==0.3]['shift est'], 'o', markersize=MS)
# pyplot.plot(shifts[df['sigma']==0.7]['shift'],shifts[df['sigma']==0.7]['shift est'], '.', markersize=MS)
# pyplot.plot(shifts[df['sigma']==0.9]['shift'],shifts[df['sigma']==0.9]['shift est'], '+', markersize=MS)
# pyplot.plot([0,2],[0,2])
# pyplot.xlabel('Shift in simulation')
# pyplot.ylabel('Calibrated shift')
# pyplot.legend(['Noise variance = 0.1','Noise variance = 0.3','Noise variance = 0.7','Noise variance = 0.9'])
# matplotlib.rc('font', size=21)
# 
x_train = np.zeros([1500, 14])
for i in range(1500):
    x_train[i] = np.append(df['lambda_gr'].iloc[i - i % 13 : i - i % 13 + 13].values, df['sparsity'].iloc[i])

df['ex-gr'] = 1
df['ex-gr'].loc[df['lambda_ex'] - df['lambda_gr']>0.008] = -1

y_t = df['ex-gr'].iloc[:1500].values

from sklearn.model_selection import train_test_split
X=df[:k]
X['lambda_1']=df['lambda_ex'].iloc[[np.floor(x/13)*13 for x in range(0,k)]].values
X['lambda_1']=df['lambda_ex'].iloc[[np.floor(x/13)*13 for x in range(0,k)]].values
X['lambda_N']=df['lambda_ex'].iloc[[np.floor(x/13)*13+12 for x in range(0,k)]].values
X=X.drop(['sigma', 'size', 'noise std est', 'shift', 'shift est','norm', 'norm_2','lambda_gr', 'lambda_ex_Atrue', 'lambda_ex_2', 'lambda_gr_2'], axis='columns')

Y=[(lambda x : -1 if df['lambda_ex'][x]-df['lambda_gr'][x]>0.01 else 1)(x) for x in range(0,k)]
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_t, test_size=0.2)
from sklearn.svm import SVC
model=SVC()
model.fit(X_train, Y_train)
model.score(X_test, Y_test)


f, ax = pl.subplots(2,2)
ax1, ax2, ax3, ax4 = ax.flatten()


spar = 4
wrong = 1
for i in range(k):
    if (df['sparsity'].iloc[i] == spar) & (df['ex-gr'].iloc[i] == -1):
        wrong = -1
    elif df['sparsity'].iloc[i] == 1:
        df['ex-gr'].iloc[i] = wrong
        wrong = 1

wrong = 1
for i in range(k-1, -1, -1):
    if (df['sparsity'].iloc[i] == spar) & (df['ex-gr'].iloc[i] == -1):
        wrong = -1
    elif df['sparsity'].iloc[i] == 13:
        df['ex-gr'].iloc[i] = wrong
        wrong = 1

# GRwrong = df.loc[(df['sparsity'] == spar) & (df['ex-gr'] == -1)]['lambda_gr']
# GRgood = df.loc[(df['sparsity'] == spar) & (df['ex-gr'] == 1)]['lambda_gr']
# SPNwrong = df.loc[(df['sparsity'] == 13) & (df['ex-gr'] == -1)]['lambda_gr']
# SPNgood = df.loc[(df['sparsity'] == 13) & (df['ex-gr'] == 1)]['lambda_gr']
# SP1wrong = df.loc[(df['sparsity'] == 1) & (df['ex-gr'] == -1)]['lambda_gr']
# SP1good = df.loc[(df['sparsity'] == 1) & (df['ex-gr'] == 1)]['lambda_gr']
# SP1good = SP1good[:len(SPNgood)]

GRwrong = df.loc[(df['sparsity'] == spar) & (df['ex-gr'] == -1)]['trace']
GRgood = df.loc[(df['sparsity'] == spar) & (df['ex-gr'] == 1)]['trace']
SPNwrong = df.loc[(df['sparsity'] == 13) & (df['ex-gr'] == -1)]['lambda_gr']
SPNgood = df.loc[(df['sparsity'] == 13) & (df['ex-gr'] == 1)]['lambda_gr']
SP1wrong = df.loc[(df['sparsity'] == 1) & (df['ex-gr'] == -1)]['lambda_gr']
SP1good = df.loc[(df['sparsity'] == 1) & (df['ex-gr'] == 1)]['lambda_gr']
SP1good = SP1good[:len(SPNgood)]
# 
# ax1.plot((13**2+(SPNwrong.values-SP1wrong.values)**2)**(1/2),GRwrong, marker='_', linestyle='None')
# ax1.plot((13**2+(SPNgood.values-SP1good.values)**2)**(1/2),GRgood, marker='+', linestyle='None')
# 
# ax6.plot((SPNwrong.values-SP1wrong.values),(GRwrong.values-SP1wrong.values)/(SPNwrong.values-SP1wrong.values), marker='_', linestyle='None')
# ax6.plot((SPNgood.values-SP1good.values),(GRgood.values-SP1good.values)/(SPNgood.values-SP1good.values), marker='+', linestyle='None')
# 
# pl.plot((SPNwrong.values-SP1wrong.values),(GRwrong.values-SP1wrong.values)/(SPNwrong.values-SP1wrong.values), marker='_', linestyle='None')
# pl.plot((SPNgood.values-SP1good.values),(GRgood.values-SP1good.values)/(SPNgood.values-SP1good.values), marker='+', linestyle='None')
# pl.plot((SPNwrong.values-SP1wrong.values),(GRwrong.values-SP1wrong.values)/(SPNwrong.values-SP1wrong.values), marker='_', linestyle='None')
# pl.plot((SPNgood.values-SP1good.values),(GRgood.values-SP1good.values)/(SPNgood.values-SP1good.values), marker='+', linestyle='None')
# pl.plot((SPNwrong.values-SP1wrong.values),(GRwrong.values-SP1wrong.values)/(SPNwrong.values-SP1wrong.values), marker='_', linestyle='None')
# pl.plot((SPNgood.values-SP1good.values),(GRgood.values-SP1good.values)/(SPNgood.values-SP1good.values), marker='+', linestyle='None')
# 
# pl.clf()
# 
# pl.plot(13/2*(SPNwrong.values-SP1wrong.values),8/2*(GRwrong.values-SP1wrong.values), marker='_', linestyle='None')
# pl.plot(13/2*(SPNgood.values-SP1good.values),8/2*(GRgood.values-SP1good.values), marker='+', linestyle='None')
# 
# pl.plot((13**2+(SPNwrong.values-SP1wrong.values)**2)**(1/2),(spar**2+(GRwrong.values-SP1wrong.values)**2)**(1/2), marker='_', linestyle='None')
# pl.plot((13**2+(SPNgood.values-SP1good.values)**2)**(1/2),(spar**2+(GRgood.values-SP1good.values)**2)**(1/2), marker='+', linestyle='None')
# 



