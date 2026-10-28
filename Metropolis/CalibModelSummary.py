import numpy as np
from os import listdir
from os.path import isfile, join, splitext
from matplotlib import pyplot as pl


def summary(mypath):
    models = ['regular', 'diff', 'log', 'logdiff']
    colorcodes = ['bx', 'gx', 'yx', 'rx']
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    mincounter = np.zeros((100, 2))
    minimas = {"regular": np.zeros((1, 2)),
               "diff": np.zeros((1, 2)),
               "log": np.zeros((1, 2)),
               "logdiff": np.zeros((1, 2))}

    for file in onlyfiles:
        errordata = np.loadtxt(mypath+file, delimiter=',')
        m_size, start = [float(t) for t in splitext(file)[0].split("_")[-2:]]
        for i in range(0, errordata.shape[1], 2):
            mincoord = np.argwhere(errordata[:, i:i + 2] == np.min(errordata[:, i:i + 2]))
            mincounter[mincoord[0][0], mincoord[0][1]] += 1
            minimas[models[mincoord[0][0]]]=np.vstack((minimas[models[mincoord[0][0]]], np.array([start, np.floor(i/2)*20])))
        # for ix, model in enumerate(models):
        #     pl.subplot(2, 2, ix+1)
        #     pl.plot(errordata[ix, ::2], errordata[ix, 1::2], colorcodes[ix], label=model)
    for ix, model in enumerate(models):
        pl.subplot(2, 2, ix+1)
        pl.plot(minimas[model][:, 0], minimas[model][:, 1], colorcodes[ix], label=model)
    # for ix, model in enumerate(models):
    #     pl.subplot(2,2,ix+1)
    #     pl.plot([1000, 12000], [1000, 12000])
    #     pl.xlabel('MSE w/o shift')
    #     pl.ylabel('MSE w/ shift')
    #     pl.title(model)
        # print(file)
        # if np.any(errordata > 3e4):
        #     print(file)
    # pl.legend()
    mincounter /= np.sum(mincounter)
    pl.show()
    pl.savefig()
    print(errordata)



if __name__ == '__main__':
    summarypath = 'E:\\PhD\\Metropolis\\Calibration_test_methods_2\\'
    summary(summarypath)