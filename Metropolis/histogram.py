import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import glob

mypath = 'E:\\PhD\\Metropolis\\TradeTest\\OnlineLSTMwithVARrange\\Total\\'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

li = []
for file in onlyfiles:
    print(file)
    content = pd.read_csv(mypath + file, sep='\t', header=None)
    li.append(content)
    fileDetails = file.split("_")
content = pd.concat(li, axis=0, ignore_index=True)
# content[1] = np.exp(content[1] + np.log(10000)) - 10000
print(str(np.mean(content[1]))+"/"+str(np.std(content[1])))
print(str(np.mean(content[1][content[1]<0])))
np.minimum(np.maximum(content[1], -1550), 1400).plot.hist(bins=35, rwidth=0.9, range=(-1550, 1400))
# content[1].plot.hist(bins=30, rwidth=0.9)
# plt.title('Profit histogram')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Profit [$]',fontsize=18)
plt.ylabel('Counts',fontsize=18)
# plt.legend(['No deviation', 'Deviation'])
plt.title('Distribution of profit on real data', fontsize=22)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)
# plt.show()
splitted = file.split("_")
# savefile = '..\\pictures\\profit_histogram_population{popultation}_sparsity{sparsity}_Tid{Tid}.png'.format(popultation=splitted[1], sparsity=splitted[2], Tid=splitted[3])
plt.savefig('E:\\PhD_\\profit_histogram_Trade_test_OnlineLSTwithVarRange_Total.png')
plt.clf()