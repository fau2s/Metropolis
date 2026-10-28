from functools import partial

import numpy as np
import optuna
from matplotlib import pyplot as pl
from scipy.fft import fft, ifft, fftshift
from scipy.signal.windows import gaussian

import LuongAttention
import RealMeanRevertingPortfolioData
import hyperparamOptimization as hypOp
from NBEATS import NBEATSx, NBEATS
from NHiTS import NHITS
from SlidingFFT import sliding_fft
from pyinform.transferentropy import transfer_entropy


def discretize_quantile(x, bins=10):

    edges = np.quantile(x, np.linspace(0,1,bins+1))

    states = np.digitize(x, edges) - 1

    states[states==bins] = bins-1

    return states

def transfer_entropy_matrix(X, bins=10):

    T,N = X.shape

    Xd = np.zeros_like(X, dtype=int)

    for i in range(N):
        Xd[:,i] = discretize_quantile(X[:,i], bins)

    TE = np.zeros((N,N))

    for i in range(N):
        for j in range(N):

            if i != j:

                TE[i,j] = transfer_entropy(
                    Xd[:,i],
                    Xd[:,j],
                    k=1
                )
    return TE

def rmt_filter(X):

    T, N = X.shape

    C = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0)
    C[-1, -1] = 1

    eigvals, eigvecs = np.linalg.eigh(C)

    q = N / T
    lambda_max = (1 + np.sqrt(q))**2

    noise_mask = eigvals < lambda_max

    avg_noise = np.mean(eigvals[noise_mask])

    eigvals_filtered = np.where(eigvals > lambda_max, eigvals, avg_noise)

    C_filtered = eigvecs @ np.diag(eigvals_filtered) @ eigvecs.T

    return C_filtered

def findBestParam():
    meanrevPortfolio = RealMeanRevertingPortfolioData.RealMeanRevertingPortfolioData()
    # tradingdata = meanrevPortfolio.getRealTradingData()
    tradingdata = meanrevPortfolio.getCryptoData()
    tradingdata = np.array(tradingdata[2,:]) # the third is bitcoin
    # tradingdata = np.diff(tradingdata)
    tradingdata = np.diff(np.log(tradingdata))

    filteredPortfolio = fft(tradingdata)
    filteredPortfolio = fftshift(filteredPortfolio) * gaussian(tradingdata.shape[0], 50)
    filteredPortfolio = np.real(ifft(fftshift(filteredPortfolio)))
    filteredPortfolio[-1:] = tradingdata[-1:]

    # optimization_function = partial(
    #     hypOp.optimize_optuna_NHITS,
    #     testdata=tradingdata[1200:],
    #     truncation_test_len=40,  # generalize to any length
    #     ratio=0.6
    # )
    # study = optuna.create_study(direction="minimize")
    #
    # study.optimize(optimization_function, n_trials=15)
    # bestparams = study.best_params  # study.bestNLPPred not to recalibrate

    lookBack = 5
    # windowSize = 100
    for windowSize in [2 ** n for n in range(6, 19)]:  # range(64, 400, 100):
        # for regressionP in range(350, 1000 - windowSize, 50):
            spectra, freqs = sliding_fft(tradingdata[:1000], windowSize, 1)
            spectraFull, _ = sliding_fft(tradingdata[900:], windowSize, 1)

            # parameters
            L = 256  # FFT window length (samples)
            step = 1  # sliding step (samples)
            in_len = 550  # number of windows used by encoder
            out_len = 30  # K: forecast horizon in windows
            latent_dim = 512
            batch_size = 128
            epochs = 200

            a = np.concatenate((np.real(spectraFull)[:, 1:int(windowSize / 2 + 1)],
                                np.imag(spectraFull)[:, 1:int(windowSize / 2 + 1)]), 1)
            filteredCorr = rmt_filter(a)
            transferEntropy = transfer_entropy_matrix(a)
            optimization_function = partial(
                hypOp.optimize_optuna_NHITS,
                testdata=a,
                truncation_test_len=windowSize,  # generalize to a ny length
                ratio=0.6  
            )
            study = optuna.create_study(direction="minimize")


            study.optimize(optimization_function, n_trials=15)
            bestparams = study.best_params  # study.bestNLPPred not to recalibrate
            # steps = 50
            # start = np.max([len(spectra), regressionP])

            pl.clf()
            for p in range(1, 2): #spectraFull[0].shape[0]
                pl.plot([spectraFull[i][p].real for i in range(len(spectraFull))])
                pl.plot([spectraFull[i][p].imag for i in range(len(spectraFull))])



            # nlpLSTMPred = LuongAttention.NLPwithLSTM(
            #     data=a,
            #     train_ratio=0.8,
            #     n_hidden=64,
            #     momentum=0.25,
            #     epc=150,
            #     learning_rate=0.00005,
            #     batch_size=batch_size,
            #     train_len=150,
            #     test_len=windowSize,
            #     attention="Luong"
            # )
            # nlpLSTMPred.constructnetwork()
            # nlpLSTMPred.trainnetwork()
            #
            # train_mae = nlpLSTMPred.train_mae
            # valid_mae = nlpLSTMPred.valid_mae
            # valid_mae_shifted = nlpLSTMPred.valid_mae_shifted
            # nlpLSTMPred.plotPredict(
            #     save=True,
            #     spar=11,
            #     start=1,
            #     length=len(spectraFull),
            #     inputLen=windowSize,
            #     layers=1,
            #     nHidden=128,
            #     train_len=150
            # )

    optimization_function = partial(
        hypOp.optimize_optuna,
        testdata=tradingdata,
        truncation_test_len=30,  # generalize to any length
        ratio=0.8
    )
    study = optuna.create_study(direction="minimize")

    study.optimize(optimization_function, n_trials=55)
    bestparams = study.best_params  # study.bestNLPPred not to recalibrate

    return bestparams


if __name__ == '__main__':
    findBestParam()
