import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm

dataset = sm.datasets.macrodata.load_pandas()
dataset = dataset.data[["realgdp", "reallcons"]].pct_change().dropna()

maxlag=12
test = 'ssr-chi2test'
def grangers_causality_matrix(X_train, variables, test = 'ssr_chi2test', verbose=False):
    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(X_train[[r,c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose:
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            dataset.loc[r,c] = min_p_value
            dataset.columns = [var + '_x' for var in variables]
            dataset.index = [var + '_y' for var in variables]
    return dataset

grangers_causality_matrix(dataset, variables = dataset.columns)