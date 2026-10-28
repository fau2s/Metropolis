import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    fileName = "D:\\Phd\\Metropolis\\S&P 500 Historical Data (1).csv"
    SP500 = pd.read_csv(fileName)
    content = SP500['Price'].str.replace(',', '').astype('float').diff(periods=-100)
    print(str(np.mean(content)) + "/" + str(np.std(content)))
    print(str(np.mean(content[content < 0])))
    np.minimum(np.maximum(content, -1550), 1400).plot.hist(bins=35, rwidth=0.9, range=(-1550, 1400))
    # content[1].plot.hist(bins=30, rwidth=0.9)
    # plt.title('Profit histogram')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Profit [$]', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    # plt.legend(['No deviation', 'Deviation'])
    plt.title('Distribution of profit on real data', fontsize=22)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    # plt.show()
    # savefile = '..\\pictures\\profit_histogram_population{popultation}_sparsity{sparsity}_Tid{Tid}.png'.format(popultation=splitted[1], sparsity=splitted[2], Tid=splitted[3])
    plt.savefig('E:\\PhD_\\profit_histogram_Trade_test_SP500_Total.png')
    plt.clf()


if __name__ == '__main__':
    main()