import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('perceptron/perc3.txt')
plt.figure(figsize=(20, 10), dpi=300)

plt.scatter(data[data[:, 0] == 0][:, 1], data[data[:, 0] == 0][:, 2], color='red')
plt.scatter(data[data[:, 0] == 1][:, 1], data[data[:, 0] == 1][:, 2], color='orange')

from sklearn import ensemble as ens
from sklearn import metrics as met
from sklearn import linear_model as lin


tr_size = 200

idx = np.random.permutation(len(data))

train = data[idx[:tr_size]]
test = data[idx[tr_size:]]
cf = lin.LogisticRegression(solver='newton-cg')
cf.fit(train[:, 1:], train[:, 0])
pred = cf.decision_function(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred)
print(auc)
cf = ens.AdaBoostClassifier()
cf.fit(train[:, 1:], train[:, 0])
pred = cf.decision_function(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred)
print(auc)
cf = ens.AdaBoostClassifier(n_estimators=1000)
cf.fit(train[:, 1:], train[:, 0])
pred = cf.decision_function(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred)
print(auc)
cf = ens.AdaBoostClassifier(n_estimators=10)
cf.fit(train[:, 1:], train[:, 0])
pred = cf.decision_function(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred)
print(auc)
cf = ens.RandomForestClassifier()
cf.fit(train[:, 1:], train[:, 0])
pred = cf.predict_proba(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred[:, 1])
print(auc)
cf = ens.RandomForestClassifier(n_estimators=1000)
cf.fit(train[:, 1:], train[:, 0])
pred = cf.predict_proba(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred[:, 1])
print(auc)
cf = ens.RandomForestClassifier(n_estimators=10)
cf.fit(train[:, 1:], train[:, 0])
pred = cf.predict_proba(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred[:, 1])
print(auc)
cf = ens.GradientBoostingClassifier()
cf.fit(train[:, 1:], train[:, 0])
pred = cf.predict_proba(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred[:, 1])
print(auc)
cf = ens.GradientBoostingClassifier(n_estimators=1000)
cf.fit(train[:, 1:], train[:, 0])
pred = cf.predict_proba(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred[:, 1])
print(auc)
cf = ens.GradientBoostingClassifier(n_estimators=10)
cf.fit(train[:, 1:], train[:, 0])
pred = cf.predict_proba(test[:, 1:])
auc = met.roc_auc_score(test[:, 0], pred[:, 1])
print(auc)


class LogisticRegression:

    def __init__(self, dim):
        pass
    # ... parameters?
    def predict(self, x):
        pass
    # ...confidence?
    def update(self, x, y, bias=True):
        pass

# ...gradient?

class LogisticRegression:

    def __init__(self, dim):
        self.dim_in = dim
        self.w = np.zeros((dim)).astype(np.float32)
        self.b = 0
        self.dw = np.zeros((dim)).astype(np.float32)
        self.db = 0
        self.alpha = 0.1

    def predict(self, x):
        return 1.0 / (1.0 + np.exp(-1 * (self.w.dot(x) + self.b)))

    def update(self, x, y, bias=True):
        n_batch = len(x)
        self.dw = (y[0] - self.predict(x[0, :])) * x[0, :] / n_batch
        if bias == True:
            self.db = (y[0] - self.predict(x[0, :])) / n_batch

        for act in range(1, n_batch):
            self.dw += (y[act] - self.predict(x[act, :])) * x[act, :] / n_batch
            if bias == True:
                self.db += (y[act] - self.predict(x[act, :])) / n_batch

        self.w += self.dw * self.alpha
        if bias == True:
            self.b += self.db * self.alpha


logreg = LogisticRegression()  # ?
logreg = LogisticRegression(len(data[0, 1:]))
batchsize = 100
pred_score = np.zeros(len(data))
pred_act = np.zeros(len(data))
n_epochs = 100

for epoch in range(n_epochs):
    for i in range(0, len(data) - 1, batchsize):
        logreg.update(data[i:i + batchsize, 1:], data[i:i + batchsize, 0])
    # ...eval: auc, accuracy...
batchsize = 100
pred_score = np.zeros((data[:, 0].size))
pred_act = np.zeros((data[:, 0].size))
n_epochs = 100

for epoch in range(n_epochs):
    for i in range(0, data[:, 0].size - 1, batchsize):
        logreg.update(data[i:i + batchsize, 1:], data[i:i + batchsize, 0])
    acc = 0
    for j in range(data[:, 0].size):
        pred_score[j] = logreg.predict(data[j, 1:])
        if (data[j, 0] == 1) & (pred_score[j] >= 0.5):
            acc += 1
        if (data[j, 0] == 0) & (pred_score[j] < 0.5):
            acc += 1
        if pred_score[j] >= 0.5:
            pred_act[j] = 1
        else:
            pred_act[j] = 0
    if epoch == n_epochs - 1:
        print(epoch)
        print(float(acc) / data[:, 0].size)

plt.figure(figsize=(20, 10), dpi=300)
# ...plot hyperplane
plt.show()
plt.figure(figsize=(20, 10), dpi=300)

x = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 1000)
if logreg.w[1] != 0:
    y = (-logreg.w[0] * x - logreg.b) / logreg.w[1]
else:
    y = np.zeros(x.size)
plt.plot(x, y)
plt.scatter(data[pred_act[:] == 0][:, 1], data[pred_act[:] == 0][:, 2], color='red')
plt.scatter(data[pred_act[:] == 1][:, 1], data[pred_act[:] == 1][:, 2], color='orange')

plt.show()

auc = np.zeros(100)
for i in range(100):
    cf = ens.AdaBoostClassifier(n_estimators=(i + 1) * 10)
    cf.fit(train[:, 1:], train[:, 0])
    pred = cf.decision_function(test[:, 1:])
    auc[i] = met.roc_auc_score(test[:, 0], pred)
    plt.figure(figsize=(20, 10), dpi=300)
plt.plot(auc, color='orange')
plt.show()
