from functools import partial
from os import listdir
from os.path import isfile, join

import numpy as np
import optuna

from Metropolis import LuongAttention, hyperparamOptimization as hypOp

testPath = '.\\NLPTest\\'
files = [f for f in listdir(testPath) if isfile(join(testPath, f))]

numberOfTrials = 5
testLength = 20

MAE = np.zeros([len(files), 10])

for count, file in enumerate(files):
    testdata = np.loadtxt(join(testPath, file))
    namesplit = file.split('_')
    MAE[count, 0] = int(''.join([n for n in namesplit[1] if n.isdigit()]))
    MAE[count, 1] = int(''.join([n for n in namesplit[2] if n.isdigit()]))
    MAE[count, 2] = int(''.join([n for n in namesplit[3] if n.isdigit()]))

    optimization_function = partial(hypOp.optimize_optuna, testdata=testdata, truncation_test_len=testLength, ratio=0.8)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function, n_trials=numberOfTrials)
    bestparams = study.best_params

    MAE[count, 3] = bestparams['n_hidden']
    MAE[count, 4] = bestparams['momentum']
    MAE[count, 5] = bestparams['epc']
    MAE[count, 6] = bestparams['learning_rate']
    MAE[count, 7] = bestparams['batch_size']
    MAE[count, 8] = bestparams['trunc_train_len']
    MAE[count, 9] = study.best_value

    nlptest = LuongAttention.NLPwithLSTM(
        testdata,
        train_ratio=0.8,
        n_hidden=bestparams['n_hidden'],
        momentum=bestparams['momentum'],
        epc=bestparams['epc'],
        learning_rate=bestparams['learning_rate'],
        batch_size=bestparams['batch_size'],
        train_len=bestparams['trunc_train_len'],
        test_len=testLength
    )
    nlptest.constructnetwork()
    nlptest.trainnetwork()
    nlptest.plotPredict(save=True, spar=MAE[count, 0], start=MAE[count, 1], length=MAE[count, 2])

    np.savetxt('NLPtestResults.txt', MAE, fmt='%.4e')