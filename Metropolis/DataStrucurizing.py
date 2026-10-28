import pandas as pd
import numpy as np
import os


def unpackData(price_data, ticker):
    if (len(price_data['prices'])):
        stock_close = [[close['formatted_date'], close['close']] for i, close in enumerate(price_data['prices']) if 'close' in close.keys()]
        if len(stock_close):
            pd.DataFrame(stock_close).to_csv('E:\\PhD\\Metropolis\\RealData\\' + ticker + '.csv')
            return len(stock_close)


def systemizing(SP500tickers, stockDataFile, realdataPath, SP500path):
    SP500TickerList = pd.read_csv(SP500tickers, header=None)
    # stockData = pd.read_csv(stockDataFile).set_index('Date')
    stockData = pd.read_csv(stockDataFile).set_index('Datetime')
    usedTicker = pd.DataFrame(columns=['Ticker', 'Size'])
    if not os.path.exists(realdataPath):
        os.makedirs(realdataPath)
    if not os.path.exists(SP500path):
        os.makedirs(SP500path)

    # iterate over each symbol
    for stockName in stockData.columns:
        stockData[stockName] = interpolate_two_consecutive_zeros(stockData[stockName])
        newDataFrame = pd.DataFrame([{
            'Ticker': stockName + '.csv',
            'Size': len(stockData[stockName][stockData[stockName] > 0])
        }])
        usedTicker = pd.concat([usedTicker, newDataFrame], ignore_index=True)
        stockData[stockName].to_csv(realdataPath + stockName + '.csv', index=True, header=False)
        if stockName in SP500TickerList[0].to_list():
            stockData[stockName].to_csv(SP500path + stockName + '.csv', index=True, header=False)
    pd.DataFrame(usedTicker).to_csv('cryptoList.txt', index=True)
    # pd.DataFrame(usedTicker).to_csv('SP500TickerList.txt', index=True)


# Function to handle interpolation for exactly 2 consecutive 0's
def interpolate_two_consecutive_zeros(series):
    zeros_count = 0
    mask = np.zeros(len(series), dtype=bool)  # Create a mask to mark values for interpolation

    for i in range(len(series)):
        if series[i] == 0:
            zeros_count += 1
        else:
            if zeros_count <= 2:
                mask[i - zeros_count:i] = True  # Mark these two values for interpolation
            zeros_count = 0

    # Mask larger sequences of zeros and leave them untouched
    series_interpolated = series.copy()
    series_interpolated[mask] = np.nan  # Mark positions for interpolation
    series_interpolated.interpolate(method='linear', inplace=True)  # Interpolate only marked positions
    if series[0] == 0.0:
        series[0] = series[1]
    return series_interpolated.fillna(series)  # Use original values where not interpolated



if __name__ == '__main__':
    SP500tickers = 'D:\\PhD_data\\SP500Tickers.csv'
    stockDataFile = 'latestStocksDataSP500.csv'
    cryptoDataFile = 'cryptoData.csv'
    realdataPath = './RealData9/'
    SP500path = './SP500_9/'
    # stocks = systemizing(SP500tickers, stockDataFile, realdataPath, SP500path)
    stocks = systemizing(SP500tickers, cryptoDataFile, realdataPath, SP500path)

