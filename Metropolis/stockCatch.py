from yahoofinancials import YahooFinancials
import pandas as pd

def unpackData(price_data, ticker):
    # price_data = YahooFinancials(ticker).get_historical_price_data("2016-09-10", "2020-01-22", "daily")
    # if ('prices' in price_data[ticker].keys()):
    if (len(price_data['prices'])):
        # if ('close' in price_data[ticker]['prices'][0].keys()):
        # print(ticker, len([close['close'] for i, close in enumerate(price_data[ticker]['prices']) if 'close' in close.keys()]))
        # print(index/length)
        stock_close = [[close['formatted_date'], close['close']] for i, close in enumerate(price_data['prices']) if 'close' in close.keys()]
        if len(stock_close):
            pd.DataFrame(stock_close).to_csv('E:\\PhD\\Metropolis\\RealData\\' + ticker + '.csv')
            return len(stock_close)

fullTickerList = pd.read_excel('E:\PhD_\Yahoo Ticker Symbols - September 2017.xlsx', 'Stock', header=3, usecols=3)

usedTicker = pd.DataFrame(['Ticker', 'Size'])

stepSize = 250
tickRange = range(0, len(fullTickerList['Ticker']), stepSize)

for i, r in enumerate(tickRange[:-1]):
    try:
        price_data = YahooFinancials(fullTickerList['Ticker'][r:tickRange[i+1]]).get_historical_price_data("2016-09-10", "2020-01-22", "daily")
    except:
        print('Nem sikerült!')
        continue
    for v in price_data:
        if 'prices' in price_data[v].keys():
            usedTicker.append({'Ticker': v, 'Size': unpackData(price_data[v], v)}, ignore_index=True)

pd.DataFrame(usedTicker).to_csv('trueTickerList.txt')
