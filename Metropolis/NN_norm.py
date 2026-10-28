import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

v = np.random.random((3))
m = np.random.random((2, 3))

v.dot(m.T)
hist = pd.read_csv('NN_data/image_histograms.txt', sep=' ', header=None)
act = pd.read_csv('NN_data/sample_histogram.txt', sep=' ', header=None)
dist = np.zeros((len(hist.index)))
dist_norm = np.zeros((len(hist.index)))
hist_vecs = np.array(hist.values[:, 1:]).astype(np.float32)
hist_vecs_norm = np.copy(hist_vecs).astype(np.float32)

act_vec = np.array(act.values[:, 1:]).astype(np.float32)
act_vec_norm = act_vec / np.linalg.norm(act_vec).astype(np.float32)
for i in range(hist_vecs[:, 0].size):
    norm = np.linalg.norm(hist_vecs[i])
    hist_vecs_norm[i] = hist_vecs[i] / norm
for i in range(hist_vecs[:, 0].size):
    dist[i] = np.linalg.norm(hist_vecs[i] - act_vec)
    dist_norm[i] = np.linalg.norm(hist_vecs_norm[i] - act_vec_norm)
top = np.argmin(dist)
top_val = np.min(dist)
top_norm = np.argmin(dist_norm)
top_norm_val = np.min(dist_norm)
print('before normalization: %s,%s %f' % (act.values[0, 0], hist.values[top, 0], top_val))
print('after normalization: %s,%s %f' % (act.values[0, 0], hist.values[top_norm, 0], top_norm_val))
img1 = plt.imread(str('NN_data/images/' + act.values[0, 0]))
plt.imshow(img1)
plt.show()
img1 = plt.imread(str('NN_data/images/' + hist.values[top, 0]))
plt.imshow(img1)
plt.show()
img1 = plt.imread(str('NN_data/images/' + hist.values[top_norm, 0]))
plt.imshow(img1)
plt.show()



