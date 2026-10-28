import pandas as pd
import yfinance as yf
import datetime
import requests
import io


def tickerList(tlType):
    if tlType == 'Nasdaq':
        url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
        s = requests.get(url).content
        companies = pd.read_csv(io.StringIO(s.decode('utf-8')))
        Symbols = companies['Symbol'].tolist()
    elif tlType == 'SP500':
        Symbols = pd.read_csv("D:\\PhD_data\\SP500Tickers.csv", header=None)[0].to_list()
    elif tlType == 'SP500CommodEtc':
        Symbols = pd.read_csv("D:\\PhD_data\\SP500CommoditiesFXETRTickers.csv", header=None)[0].to_list()

    return Symbols


def download():
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2026, 4, 30)

    Symbols = tickerList('SP500')
    # Symbols = tickerList('SP500CommodEtc')
    # Symbols = tickerList('Nasdaq')

    # create empty dataframe
    stock_final = pd.DataFrame()
    # iterate over each symbol
    # for i in Symbols:
    #
    #     # print the symbol which is being downloaded
    #     print(str(Symbols.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)
    #
    #     try:
    #         # download the stock price
    #         stock = []
    #         stock = yf.download(["AAPL", "GOOG", "CCL"], start=start, end=end, progress=True, threads=False)
    #
    #         # append the individual stock prices
    #         if len(stock) == 0:
    #             None
    #         else:
    #             stock['Name'] = i
    #             stock_final = pd.concat([stock_final, stock], sort=False)
    #     except Exception:
    #         None
    crypto_tickers = [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "SOL-USD",
        "XRP-USD",
        "ADA-USD",
        "DOGE-USD"
    ]
    return yf.download(
        crypto_tickers,
        period="730d",
        interval="1h",
        auto_adjust=True,
        progress=False,
        group_by="ticker"
    )

    # return yf.download(Symbols, start=start, end=end, progress=True, threads=False, group_by="ticker")


if __name__ == '__main__':
    stocks = download()
    Symbols = tickerList('SP500')
    crypto_tickers = [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "SOL-USD",
        "XRP-USD",
        "ADA-USD",
        "DOGE-USD"
    ]
    # valid_tickers = [ticker for ticker in Symbols if not stocks[ticker]['Close'].isna().all()]
    # stocks[valid_tickers].swaplevel(axis=1)['Volume'].reindex(sorted(valid_tickers), axis=1).to_csv("latestStocksDataSP500Volume.csv")
    stocks[crypto_tickers].swaplevel(axis=1)['Close'].reindex(sorted(crypto_tickers), axis=1).to_csv("cryptoData.csv")
    # stocks[valid_tickers].swaplevel(axis=1)['Close'].reindex(sorted(valid_tickers), axis=1).to_csv(
    #     "latestStocksDataSP500.csv")
    # stocks.pivot_table(index='Date', values='Close', columns='Name', fill_value=0).to_csv("latestStocksDataSP500CommodEtc.csv")
    # stocks.pivot_table(index='Date', values='Close', columns='Name', fill_value=0).to_csv("latestStocksDataNasdaq.csv")
    # stocks.to_csv("newStocksData.csv")
