import numpy as np
import pandas as pd


def main(filename):
    dataMatrix = pd.read_csv(filename, sep='\t')
    histogram = dataMatrix.hist(column='1', bins=15)
    histogram.savetxt()


if __name__ == '__main__':
    main("")