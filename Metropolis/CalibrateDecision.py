import pandas as pd
# from sklearn.manifold.tests.test_t_sne import test_n_components_range
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import data
# from sklearn.svm import SVC
import ExtremeLearningMachine as elm
from matplotlib import pyplot as pypl


class CalibrateDecision:

    def __init__(self, StartSize, EndSize, iterations, isSimulated, hiddenUnits = 3, threshold = 0.008):
        self.start = StartSize
        self.n = EndSize
        self.simulated = isSimulated
        self.iterations = iterations
        self.threshold = threshold
        numberOfData = int((EndSize*(EndSize+1)/2-StartSize*(StartSize-1)/2)*iterations)
        self.df = pd.DataFrame({'size':  np.ones(numberOfData),
                            'lambda_ex': np.ones(numberOfData),
                            'lambda_gr': np.ones(numberOfData),
                            'sparsity': np.ones(numberOfData),
                            'ex-gr': np.ones(numberOfData)})

        self.hiddenUnits = hiddenUnits

    def calibrate(self, sigma):
        k = 0
        for i in range(self.start, self.n, 1):
            print('The size of the matrix is {:d}.'.format(i))
            x_train = np.zeros([self.iterations*i, i+1])
            y_t = np.zeros([self.iterations*i, 2])
            # lexmax = np.zeros(self.iterations)
            # lgrmax = np.zeros(self.iterations)
            for iter in range(self.iterations):
                testData = data.Data(n=i, L=i, simulated=False)
                for j in reversed(range(1, i + 1)):
                    self.df['sparsity'][k] = j
                    testData.L = j
                    testData.parameterEstimation()
                    self.df['size'][k] = i
                    self.df['lambda_ex'][k] = testData.solve_exhaust()[1]
                    self.df['lambda_gr'][k] = testData.solve_greedy()[1]
                    if j == i:
                        kmax = k
                        lexmax = self.df['lambda_ex'][kmax]
                        lgrmax = self.df['lambda_gr'][kmax]
                    self.df['lambda_ex'][k] /= lexmax
                    self.df['lambda_gr'][k] /= lgrmax
                    # x_train[i*iter+j-1, :] = [self.df['lambda_gr'][k], j]
                    x_train[iter*i:i*(iter+1), j-1] = self.df['lambda_gr'][k]
                    x_train[iter*i+j-1, i] = self.df['lambda_gr'][k]
                    if self.df['lambda_ex'][k] - self.df['lambda_gr'][k] > self.threshold:
                        print([np.linalg.matrix_rank(testData.A), self.threshold])
                        self.df['ex-gr'][k] = -1
                        y_t[i*iter+j-1, 1] = 1
                    else:
                        y_t[i * iter + j - 1, 0] = 1
                        # self.df['ex-gr'] = 1
                    k += 1
                    print(np.linalg.matrix_rank(testData.A))
            pd.DataFrame({'size': [i], 'divergence': [len(self.df[(self.df['ex-gr'] == -1) & (self.df['size'] == i)])]}).to_csv('E:\\PhD\\Metropolis\\Greedy_Exhaustive_divergence.csv', header=False, index=False, mode='a')
            print(len(self.df[self.df['ex-gr'] == -1]))
            pca = PCA(n_components=2)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            pypl.clf()
            pypl.plot(x_train[y_t[:, 0] == 1, 0], x_train[y_t[:, 0] == 1, 1], 'xb')
            pypl.plot(x_train[y_t[:, 1] == 1, 0], x_train[y_t[:, 1] == 1, 1], 'xg')
            pypl.xlabel('First principal component')
            pypl.ylabel('Second principal component')
            pypl.legend(['No deviation', 'Deviation'])
            pypl.title('First two principal components with explained variance {:01.4f}'.format(np.sum(pca.explained_variance_ratio_[:2])) )
            pypl.savefig('..\\pictures\\PCA_deviations_realdata_N{i}_iter{iter}.png'.format(i=i, iter=self.iterations))
            X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_t, test_size=0.3)
            Win, Wout = elm.classification(X_train, Y_train, X_test, Y_test, self.hiddenUnits)
            np.save('E:\\PhD\\Metropolis\\InputWeights_{0}'.format(i), Win)
            np.save('E:\\PhD\\Metropolis\\OutputWeights_{0}'.format(i), Wout)
            # return Win, Wout


if __name__ == '__main__':
    calib = CalibrateDecision(15, 18, 90, False, 12)
    calib.calibrate(0)
