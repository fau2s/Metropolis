import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble as ens
from sklearn import metrics as met
from sklearn import linear_model as lin
from sklearn import svm

from scipy import spatial as sp

data = np.genfromtxt('ring.csv', delimiter=',')

plt.figure(figsize=(20, 10), dpi=300)

plt.scatter(data[data[:, 2] == 0][:, 0], data[data[:, 2] == 0][:, 1], color='red')
plt.scatter(data[data[:, 2] == 1][:, 0], data[data[:, 2] == 1][:, 1], color='orange')

tr_size = 1000

idx = np.random.permutation(len(data))

train = data[idx[:tr_size]]
test = data[idx[tr_size:]]
cf = lin.LogisticRegression()
cf.fit(train[:, :2], train[:, 2])
pred = cf.decision_function(test[:, :2])
pred_cla = cf.predict(test[:, :2])
auc = met.roc_auc_score(test[:, 2], pred)
acc = met.accuracy_score(y_true=test[:, 2], y_pred=pred_cla)
print("auc: %f acc: %f" % (auc, acc))
cf = svm.LinearSVC()
cf.fit(train[:, :2], train[:, 2])
pred = cf.decision_function(test[:, :2])
pred_cla = cf.predict(test[:, :2])
auc = met.roc_auc_score(test[:, 2], pred)
acc = met.accuracy_score(y_true=test[:, 2], y_pred=pred_cla)
print("linear kernel auc: %f acc: %f" % (auc, acc))
cf = svm.SVC(kernel='poly', degree=2, cache_size=30000)
cf.fit(train[:, :2], train[:, 2])
pred = cf.decision_function(test[:, :2])
pred_cla = cf.predict(test[:, :2])
auc = met.roc_auc_score(test[:, 2], pred)
acc = met.accuracy_score(y_true=test[:, 2], y_pred=pred_cla)
print("poly auc: %f acc: %f" % (auc, acc))
cf = svm.SVC(kernel='rbf', degree=2, cache_size=30000)
cf.fit(train[:, :2], train[:, 2])
pred = cf.decision_function(test[:, :2])
pred_cla = cf.predict(test[:, :2])
auc = met.roc_auc_score(test[:, 2], pred)
acc = met.accuracy_score(y_true=test[:, 2], y_pred=pred_cla)
print("rbf auc: %f acc: %f" % (auc, acc))
# simple linear kernel
ker_tr = train[:, :2].dot(train[:, :2].T).astype(np.float32)
ker_te = test[:, :2].dot(train[:, :2].T).astype(np.float32)

cf = svm.SVC(kernel='precomputed', cache_size=30000)
cf.fit(ker_tr, train[:, 2])
pred = cf.decision_function(ker_te)
pred_cla = cf.predict(ker_te)
auc = met.roc_auc_score(test[:, 2], pred)
acc = met.accuracy_score(y_true=test[:, 2], y_pred=pred_cla)
print("prec linear auc: %f acc: %f" % (auc, acc))

print(np.min(pred))
plt.figure(figsize=(20, 10), dpi=300)

plt.scatter(test[pred_cla[:] == 0][:, 0], test[pred_cla[:] == 0][:, 1], color='red')
plt.scatter(test[pred_cla[:] == 1][:, 0], test[pred_cla[:] == 1][:, 1], color='orange')

plt.show()

plt.figure(figsize=(20, 10), dpi=300)

plt.hist(pred, bins=100)

plt.figure(figsize=(20, 10), dpi=300)

plt.scatter(test[pred[:] < -1.0][:, 0], test[pred[:] < -1.0][:, 1], color='red')
plt.scatter(test[pred[:] >= -1.0][:, 0], test[pred[:] >= -1.0][:, 1], color='orange')

plt.show()

# simple linear kernel
ker_tr_raw = train[:, :2].dot(train[:, :2].T).astype(np.float32)
ker_te_raw = test[:, :2].dot(train[:, :2].T).astype(np.float32)

# make it rbf, note np.exp does not broadcast!!!
ker_tr = np.power(ker_tr_raw, 2)
ker_te = np.power(ker_te_raw, 2)

cf = svm.SVC(kernel='precomputed', cache_size=30000)
cf.fit(ker_tr, train[:, 2])
pred = cf.decision_function(ker_te)
pred_cla = cf.predict(ker_te)
auc = met.roc_auc_score(test[:, 2], pred)
acc = met.accuracy_score(y_true=test[:, 2], y_pred=pred_cla)
print("prec poly auc: %f acc: %f" % (auc, acc))


def rbf_kernel(x_te, x_tr, gamma=1):
    return np.exp(-gamma * sp.distance.cdist(x_te, x_tr, metric='sqeuclidean')).astype(np.float16)


def dot_kernel(x_te, x_tr, gamma=1):
    return (gamma * x_te.dot(x_tr.T)).astype(np.float32)


def poly_kernel(x_te, x_tr, gamma=1, degree=2):
    return np.power((gamma * x_te.dot(x_tr.T)).astype(np.float32), degree)


def dot_rbf_kernel(x_te, x_tr, gamma=1):
    base = 2.0 * gamma * (x_te.astype(np.float32)).dot(x_tr.T.astype(np.float32))
    for k in range(len(base[:, 0])):
        base[k] -= gamma * (x_te[k].astype(np.float32)).dot(x_te[k].astype(np.float32))
    for k in range(len(base[0, :])):
        base[:, k] -= gamma * (x_tr[k].astype(np.float32)).dot(x_tr[k].astype(np.float32))
    # np.fill_diagonal(base,0.0)
    return np.exp(base)  # .astype(np.float32)


# Why?
def dot_rbf_kernel_tr(x_tr, gamma=1):
    base = 2.0 * gamma * (x_tr.astype(np.float32)).dot(x_tr.T.astype(np.float32))
    for k in range(len(base[:, 0])):
        act = gamma * (x_tr[k].astype(np.float32)).dot(x_tr[k].astype(np.float32))
        base[k, :] -= act
        base[:, k] -= act
    np.fill_diagonal(base, 0.0)
    return np.exp(base)  # (0.5*(base+base.T)).astype(np.float32)


x_tr = train[:, :2]
x_te = test[:, :2]

ker_tr = rbf_kernel(x_te=x_tr, x_tr=x_tr)
ker_te = rbf_kernel(x_te=x_te, x_tr=x_tr)

cf = svm.SVC(kernel='precomputed', cache_size=30000)
cf.fit(ker_tr, train[:, 2])
pred = cf.decision_function(ker_te)
pred_cla = cf.predict(ker_te)
auc = met.roc_auc_score(test[:, 2], pred)
acc = met.accuracy_score(y_true=test[:, 2], y_pred=pred_cla)
print("prec rbf auc: %f acc: %f" % (auc, acc))
x_tr = train[:, :2]
x_te = test[:, :2]

ker_tr = dot_rbf_kernel_tr(x_tr=x_tr)
ker_te = dot_rbf_kernel(x_te=x_te, x_tr=x_tr)

cf = svm.SVC(kernel='precomputed', cache_size=30000)
cf.fit(ker_tr, train[:, 2])
pred = cf.decision_function(ker_te)
pred_cla = cf.predict(ker_te)
auc = met.roc_auc_score(test[:, 2], pred)
acc = met.accuracy_score(y_true=test[:, 2], y_pred=pred_cla)
print("prec rbf auc: %f acc: %f" % (auc, acc))
x_tr = train[:, :2]
x_te = test[:, :2]

ker_tr = poly_kernel(x_tr=x_tr, x_te=x_tr)
ker_te = poly_kernel(x_te=x_te, x_tr=x_tr)

cf = svm.SVC(kernel='precomputed', cache_size=30000)
cf.fit(ker_tr, train[:, 2])
pred = cf.decision_function(ker_te)
pred_cla = cf.predict(ker_te)
auc = met.roc_auc_score(test[:, 2], pred)
acc = met.accuracy_score(y_true=test[:, 2], y_pred=pred_cla)
print("prec rbf auc: %f acc: %f" % (auc, acc))
plt.figure(figsize=(20, 10), dpi=300)

plt.scatter(test[pred_cla[:] == 0][:, 0], test[pred_cla[:] == 0][:, 1], color='red')
plt.scatter(test[pred_cla[:] == 1][:, 0], test[pred_cla[:] == 1][:, 1], color='orange')

plt.show()

plt.figure(figsize=(20, 10), dpi=300)

plt.hist(pred, bins=100)

plt.figure(figsize=(20, 10), dpi=300)

plt.scatter(test[pred[:] < -1.0][:, 0], test[pred[:] < -1.0][:, 1], color='red')
plt.scatter(test[pred[:] >= -1.0][:, 0], test[pred[:] >= -1.0][:, 1], color='orange')

plt.show()

