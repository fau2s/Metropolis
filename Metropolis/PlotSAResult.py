import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as pl





def plots(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in onlyfiles:
        table = np.genfromtxt(path + file, delimiter=',')
        pl.clf()


        fig, ((ax1, ax2)) = pl.subplots(1,2)
        ax1.plot(table[:, 0], table[:, 3], 'x')
        ax1.plot([min(table[:, 0]), max(table[:, 0])],[min(table[:, 0]), max(table[:, 0])],'r-')
        ax1.set_xlabel('Energy w/ old method')
        ax1.set_ylabel('Energy w/ new method')
        ax1.set_title('Energy comparision for the old and new method')#.format(asset=testwithshift.ticker[t][:-4]))
        # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])
        ax2.plot(table[:, 2], table[:, 5], 'x')
        ax2.plot([min(table[:, 2]), max(table[:, 2])], [min(table[:, 2]), max(table[:, 2])], 'r-')
        ax2.set_xlabel('Function calls w/ old method')
        ax2.set_ylabel('Function calls w/ new method')
        ax2.set_title('Function calls comparision for the old and new method')#.format(asset=testwithshift.ticker[t][:-4]))
        # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])
        # ax3.plot(table[:, 2], table[:, 5], 'x')
        # ax3.plot([min(table[:, 2]), max(table[:, 2])], [min(table[:, 2]), max(table[:, 2])], 'r-')
        # ax3.set_xlabel('Function calls w/ old method')
        # ax3.set_ylabel('Function calls w/ new method')
        # ax3.set_title('Function calls comparision for the old and new method')#.format(asset=testwithshift.ticker[t][:-4]))
        # # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        # #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])
        # ax4.plot(table[:, 0], table[:, 2], 'rx')
        # ax4.plot(table[:, 3], table[:, 5], 'bx')
        # ax4.set_xlabel('Energy')
        # ax4.set_ylabel('Number of function calls')
        # ax4.set_title('Value of asset {asset} vs. estimated noiseless value')#.format(asset=testwithshift.ticker[t][:-4]))
        # ax4.legend(['Old method', 'New method'])

        fig.set_size_inches(15, 10)
        fig.savefig(file[:-4] + '_2.png')
        fig.clf()


        # fig, ((ax1, ax2), (ax3, ax4)) = pl.subplots(2,2)
        # ax1.plot(table[:, 0], table[:, 3], 'x')
        # ax1.plot([min(table[:, 0]), max(table[:, 0])],[min(table[:, 0]), max(table[:, 0])],'r-')
        # ax1.set_xlabel('Energy w/ old method')
        # ax1.set_ylabel('Energy w/ new method')
        # ax1.set_title('Energy comparision for the old and new method')#.format(asset=testwithshift.ticker[t][:-4]))
        # # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        # #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])
        # ax2.plot(table[:, 1], table[:, 4], 'x')
        # ax2.plot([min(table[:, 1]), max(table[:, 1])], [min(table[:, 1]), max(table[:, 1])], 'r-')
        # ax2.set_xlabel('Running time w/ old method')
        # ax2.set_ylabel('Running time w/ new method')
        # ax2.set_title('Running time comparision for the old and new method')#.format(asset=testwithshift.ticker[t][:-4]))
        # # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        # #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])
        # ax3.plot(table[:, 2], table[:, 5], 'x')
        # ax3.plot([min(table[:, 2]), max(table[:, 2])], [min(table[:, 2]), max(table[:, 2])], 'r-')
        # ax3.set_xlabel('Function calls w/ old method')
        # ax3.set_ylabel('Function calls w/ new method')
        # ax3.set_title('Function calls comparision for the old and new method')#.format(asset=testwithshift.ticker[t][:-4]))
        # # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        # #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])
        # ax4.plot(table[:, 0], table[:, 2], 'rx')
        # ax4.plot(table[:, 3], table[:, 5], 'bx')
        # ax4.set_xlabel('Energy')
        # ax4.set_ylabel('Number of function calls')
        # ax4.set_title('Value of asset {asset} vs. estimated noiseless value')#.format(asset=testwithshift.ticker[t][:-4]))
        # ax4.legend(['Old method', 'New method'])
        #
        # fig.set_size_inches(15, 10)
        # fig.savefig(file[:-4] + '.png')
        # fig.clf()
        # pl.clf()
        # pl.xlabel('Energy w/ old method')
        # pl.ylabel('Energy w/ new method')
        # pl.title('Value of asset {asset} vs. estimated noiseless value'.format(asset=testwithshift.ticker[t][:-4]))
        # pl.plot(table[:, 0], table[:, 3], 'x')
        # pl.legend(['TS with estimated AR matrix w/ constant shift', 'TS with estimated AR matrix w/o constant shift',
        #            'Real Asset ({asset})'.format(asset=testwithshift.ticker[t][:-4])])
        # pl.savefig(realdataPath + 'Real_estimated_N{i}_{s}.png'.format(i=size, s=testwithshift.ticker[t][:-4]))
        #
        # pl.plot(table[:, 0], table[:, 3], 'x')
        #
        # pl.plot(noiselessWithOutShift[t, :30].transpose())
        # pl.plot(testwithshift.Pperf[t, :30].transpose())




if __name__ == '__main__':
    path = '.\\SATest\\'
    plots(path)


