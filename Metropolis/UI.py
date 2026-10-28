import pickle
import tkinter as tk
from random import randint
from tkinter import filedialog, Text, ttk
import os
import Trading_2, Trading, getStockData
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

root = tk.Tk()
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
editmenu = tk.Menu(menubar, tearoff=0)
helpmenu = tk.Menu(menubar, tearoff=0)

apps = []
tradings = []

if os.path.isfile("save.txt"):
    with open('save.txt', 'r') as f:
        tempApps = f.read()
        tempApps = tempApps.split(',')
        apps = [x for x in tempApps if x.strip()]

def addApp():

    for widget in frame.winfo_children():
        widget.destroy()

    filename = filedialog.askopenfilename(
        initialdir="/",
        title="Select File",
        filetypes=(("executable","*.exe"), ("all files","*.*"))
    )
    apps.append(filename)
    print(filename)
    for app in apps:
        label = tk.Label(frame, text=app, bg="gray")
        label.pack()

def runApps():
    for app in apps:
        os.startfile(app)

def runTradings():
    global dummyTrade
    dummyTrade = Trading.Trading(
        n=430,
        spars=11,
        simulated=False,
        idTimeWindow=int(inputText1.get()), #400
        tradingStart=int(inputText2.get()), #1900
        tradingTimeWindow=int(inputText3.get()), #150
        useShift=True,
        n_diff=0,
        model='log',
        sigma=1e-1
    )
    dummyTrade.calibrating()
    # dummyTrade.process()
    # trade.set(dummyTrade)
    dummyTrade.plotSimplePortfolio(figure1)
    bar1.draw()
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    dummyTrade.plotSeq2SeqTrainingResult(figure5)
    canvas5.draw()
    canvas5.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    dummyTrade.plotConstituents(figure7)
    canvas7.draw()
    canvas7.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    dummyTrade.plotFilter(figure8)
    canvas8.draw()
    canvas8.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    # rolling_return_distribution(window_size=350)
    plot_2d_distribution(bins=30)
    canvas9.draw()
    canvas9.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

    weightTable = tk.ttk.Treeview(lf3)
    weightTable['columns'] = \
        ("Stock Name exhaustive", "Weight exhaustive",
         "Stock Name greedy", "Weight greedy",
         "Stock Name annealing", "Weight annealing")
    weightTable.column("#0", width=0)
    weightTable.column("Stock Name exhaustive", anchor=tk.W, width=120)
    weightTable.column("Weight exhaustive", anchor=tk.CENTER, width=120)
    weightTable.column("Stock Name greedy", anchor=tk.W, width=120)
    weightTable.column("Weight greedy", anchor=tk.CENTER, width=120)
    weightTable.column("Stock Name annealing", anchor=tk.W, width=120)
    weightTable.column("Weight annealing", anchor=tk.CENTER, width=120)

    weightTable.heading("#0", text="", anchor=tk.W)
    weightTable.heading("Stock Name exhaustive", text="Stock Name exhaustive", anchor=tk.W)
    weightTable.heading("Weight exhaustive", text="Weight exhaustive", anchor=tk.W)
    weightTable.heading("Stock Name greedy", text="Stock Name greedy", anchor=tk.W)
    weightTable.heading("Weight greedy", text="Weight greedy", anchor=tk.W)
    weightTable.heading("Stock Name annealing", text="Stock Name annealing", anchor=tk.W)
    weightTable.heading("Weight annealing", text="Weight annealing", anchor=tk.W)

    names_exhaustive = dummyTrade.timeSeries.ticker[dummyTrade.portfolios.weights['exhaustive'][0].astype(bool)]
    weights_exhaustive = dummyTrade.portfolios.weights['exhaustive'][0][dummyTrade.portfolios.weights['exhaustive'][0] != 0]
    names_greedy = dummyTrade.timeSeries.ticker[dummyTrade.portfolios.weights['greedy'][0].astype(bool)]
    weights_greedy = dummyTrade.portfolios.xoptbuy0[dummyTrade.portfolios.xoptbuy0 != 0]
    names_annealing = dummyTrade.timeSeries.ticker[dummyTrade.portfolios.weights['annealing'][0].astype(bool)]
    weights_annealing = dummyTrade.portfolios.weights['annealing'][0][dummyTrade.portfolios.weights['annealing'][0] != 0]

    for count, data in enumerate(zip(names_exhaustive, weights_exhaustive, names_greedy, weights_greedy, names_annealing, weights_annealing)):
        weightTable.insert(parent='', index='end', iid=count, text="",
                           values=(data[0], data[1], data[2], data[3], data[4], data[5]))

    weightTable.pack()

def saveTrade():
    global dummyTrade
    with open('dummytrade_portfolios.pkl', 'wb') as outp:
        pickle.dump(dummyTrade.portfolios, outp, pickle.HIGHEST_PROTOCOL)
    with open('dummytrade_timeSeries.pkl', 'wb') as outp:
        pickle.dump(dummyTrade.timeSeries, outp, pickle.HIGHEST_PROTOCOL)
    with open('dummytrade_lstmpredictedPortfolio.pkl', 'wb') as outp:
        pickle.dump(dummyTrade.lstmpredictedPortfolio, outp, pickle.HIGHEST_PROTOCOL)
    with open('dummytrade_lstmpredictedPortfolioInitOffline.pkl', 'wb') as outp:
        pickle.dump(dummyTrade.lstmpredictedPortfolioInitOffline, outp, pickle.HIGHEST_PROTOCOL)


def loadTrade():
    global dummyTrade
    with open('dummytrade.pkl', 'rb') as inp:
        dummyTrade = pickle.load(inp)


def plotVar1():
    global dummyTrade
    if dummyTrade is None:
        return
    if var1.get():
        dummyTrade.plotTest(dummyTrade.nlpPred, figure1)
    if not var1.get():
        dummyTrade.plotSimplePortfolio(figure1)

    bar1.draw()
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


def plotOfflineLSTM():
    global dummyTrade
    if dummyTrade is None:
        return
    if var3.get():
        dummyTrade.plotPortfolio(dummyTrade.nlpPred, figure1)
    if not var3.get():
        dummyTrade.plotSimplePortfolio(figure1)

    bar1.draw()
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


def plotTradingRange():
    global dummyTrade
    if dummyTrade is None:
        return
    if var2.get():
        dummyTrade.plotTest(dummyTrade.nlpPred, figure1)
    if not var2.get():
        dummyTrade.plotSimplePortfolio(figure1)

    bar1.draw()
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


def plotPortfolioAndPrediction():
    global dummyTrade
    if dummyTrade is None:
        return
    option = [str1.get(), str2.get(), str3.get(), str4.get(), str5.get(), str6.get()]
    dummyTrade.plotPortfolioAndPredition(dummyTrade.nlpPred, figure1, option)
    bar1.draw()
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

def downloadStocks():
    getStockData.download()


def getStocks():
    print(dummyTrade.portfolios.weights)


def plotStockPrices():
    global dummyTrade
    index = int(selected.get())
    figure2.clf()
    ax = figure2.add_subplot(111)
    ax.plot(dummyTrade.timeSeries.tr_P_all[index, :])
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


def saveImage():
    realdataPath = '.\\Portfolio_test_plot.png'
    figure1.savefig(realdataPath)

def getInput():
    inputText1.get(1.0, "end-1c")


def calibrateVAR():
    global dummyTrade
    # Define ranges for tradingStart and idTimeWindow
    trading_start_values = [2000]  # Example values
    id_time_window_values = [480, 960, 1440, 1920]  # Example values
    varParams = np.arange(1, 10)
    # Create a DataFrame to store results
    column_labels = [f"T{t}_W{w}" for t in trading_start_values for w in id_time_window_values]
    results_df = pd.DataFrame(index=varParams, columns=column_labels)
    # Iterate over the values
    for trading_start in trading_start_values:
        for id_time_window in id_time_window_values:
            for varp in np.arange(1, 10):
                dummyTrade = Trading.Trading(
                    n=480,
                    spars=7,
                    simulated=False,
                    idTimeWindow=id_time_window,
                    tradingStart=trading_start,
                    tradingTimeWindow=50,
                    useShift=True,
                    n_diff=0,
                    model='log',
                    sigma=1e-1
                )
                dummyTrade.timeSeries.estA()
                dummyTrade.timeSeries.estAVarP(varp) #dummyTrade.timeSeries.estAVarP(int(inputText4.get()))
                randIndex = randint(0, 479)
                generatedFull = \
                    dummyTrade.timeSeries.generateNoiselessFrom_A(
                        100,
                        dummyTrade.timeSeries.Pid[:, -varp:].copy(),
                        varp
                    )
                generated = generatedFull[randIndex, :].transpose()
                results_df.loc[varp, f"T{trading_start}_W{id_time_window}"] = \
                    [np.sum(np.abs(generatedFull - dummyTrade.timeSeries.P_all[:, trading_start:trading_start + 100]))/480/100]
                np.savetxt('MAEofVAR{p}Length{lenght}Id{id}Start{s}'.format(p=varp, lenght=30, id=id_time_window, s=trading_start),
                           [np.sum(np.abs(generatedFull - dummyTrade.timeSeries.P_all[:, trading_start:trading_start + 100]))/480/100])
    results_df.to_csv("MAE_results_480_100 .csv")

def replotVAR():
    global dummyTrade
    randIndex = randint(0, 480)
    generatedFull = dummyTrade.timeSeries.generateNoiselessFrom_A(
            100,
            dummyTrade.timeSeries.Pid[:, -int(inputText4.get()):].copy(),
            int(inputText4.get())
        )
    generated =generatedFull[randIndex, :].transpose()
    np.savetxt('MAEofVAR{p}Length{lenght}'.format(p=int(inputText4.get()), lenght=100),
               [np.sum(np.abs(generatedFull - dummyTrade.timeSeries.P_all[:, 1900:2000]))/100/452])
    figure4.clf()
    ax1 = figure4.add_subplot(111)
    ax1.plot(
        range(100),
        generated,
        'bx-'
    )
    ax1.plot(range(100), dummyTrade.timeSeries.P_all[randIndex, 1900:2000].transpose(), 'g--')
    canvas4.draw()
    canvas4.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

def plotVAR():
    global dummyTrade
    dummyTrade = Trading.Trading(
        n=480,
        spars=11,
        simulated=False,
        idTimeWindow=400,
        tradingStart=1500,
        tradingTimeWindow=150,
        useShift=True,
        n_diff=0,
        model='log',
        sigma=1e-1
    )
    figure4.clf()
    ax1 = figure4.add_subplot(111)
    randIndex = randint(0, 480)
    ax1.plot(range(30), dummyTrade.timeSeries.P_all[randIndex, 1500:1530].transpose(), 'g-', label='Original')
    for i in range(1, int(inputText4.get())+1):
        dummyTrade.timeSeries.estA()
        dummyTrade.timeSeries.estAVarP(i)

        from statsmodels.tsa.api import VAR

        model = VAR(dummyTrade.timeSeries.Pid.transpose())
        results = model.fit(i)

        lag_order = results.k_ar
        forecast_input = dummyTrade.timeSeries.Pid[:, -lag_order:]
        forecast = results.forecast(y=forecast_input.transpose(), steps=30)

        generatedFull = \
            dummyTrade.timeSeries.generateNoiselessFrom_A(
                30,
                dummyTrade.timeSeries.Pid[:, -i:].copy(),
                i
            )
        # generated = generatedFull[randIndex, :].transpose()
        generated = forecast[:, randIndex]
        ax1.plot(
            range(30),
            generated,
            '--',
            label="Var({parameter})".format(parameter=i)
        )
        # ax1.plot(range(30), generatedOriginal, 'yo')
    ax1.legend()
    ticker = dummyTrade.timeSeries.ticker[randIndex].split('.')[0]
    title = "Zero mean log real data and VAR regressions of stock {stockIdx}".format(stockIdx=ticker)
    ax1.set_title(title)
    ax1.figure.savefig(
        'VARParameterTest_{ticker}.png'.format(ticker=ticker))
    canvas4.draw()
    canvas4.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


def plotSeq2SeqTrainResult():
    global dummyTrade
    dummyTrade.plotSeq2SeqTrainingResult(figure5, int(inputText5.get()))
    canvas5.draw()
    canvas5.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


def downPlotSeq2SeqTrainResult():
    global dummyTrade
    target = np.max((int(inputText5.get()) - 1, 0))
    inputText5.delete(0, tk.END)
    inputText5.insert(0, target)
    dummyTrade.plotSeq2SeqTrainingResult(
        figure5,
        int(inputText5.get())
    )
    canvas5.draw()
    canvas5.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


def upPlotSeq2SeqTrainResult():
    global dummyTrade
    target = np.min((int(inputText5.get()) + 1, dummyTrade.nlpLSTMPred.X_output_train.shape[1] - 1))
    inputText5.delete(0, tk.END)
    inputText5.insert(0, target)
    dummyTrade.plotSeq2SeqTrainingResult(
        figure5,
        int(inputText5.get())
    )
    canvas5.draw()
    canvas5.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
# canvas = tk.Canvas(root, height=900, width=1800, bg="#263D42")
# canvas.create_text(100, 100, text=getStocks, fill="black")
# canvas.pack()

def plotConstituents():
    global dummyTrade

def replotFFT():
    global dummyTrade
    dummyTrade.plotFilter(figure8, selectedFilter.get(), float(ratio.get()))
    canvas8.draw()
    canvas8.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

def runTrainig():
    global dummyTrade
    dummyTrade = Trading.Trading(
        n=480,
        spars=11,
        simulated=False,
        idTimeWindow=int(inputText1.get()), #400
        tradingStart=int(inputText2.get()), #1900
        tradingTimeWindow=int(inputText3.get()), #150
        useShift=True,
        n_diff=0,
        model='regular',
        sigma=1e-1
    )
    dummyTrade.calibrating()
    dummyTrade.plotSimplePortfolio(figure1)
    bar1.draw()
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    dummyTrade.plotSeq2SeqTrainingResult(figure5)
    canvas5.draw()
    canvas5.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    dummyTrade.plotConstituents(figure7)
    canvas7.draw()
    canvas7.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    dummyTrade.plotFilter(figure8)
    canvas8.draw()
    canvas8.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

    canvas9.draw()
    canvas9.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

    weightTable = tk.ttk.Treeview(lf3)
    weightTable['columns'] = \
        ("Stock Name exhaustive", "Weight exhaustive",
         "Stock Name greedy", "Weight greedy",
         "Stock Name annealing", "Weight annealing")
    weightTable.column("#0", width=0)
    weightTable.column("Stock Name exhaustive", anchor=tk.W, width=120)
    weightTable.column("Weight exhaustive", anchor=tk.CENTER, width=120)
    weightTable.column("Stock Name greedy", anchor=tk.W, width=120)
    weightTable.column("Weight greedy", anchor=tk.CENTER, width=120)
    weightTable.column("Stock Name annealing", anchor=tk.W, width=120)
    weightTable.column("Weight annealing", anchor=tk.CENTER, width=120)

    weightTable.heading("#0", text="", anchor=tk.W)
    weightTable.heading("Stock Name exhaustive", text="Stock Name exhaustive", anchor=tk.W)
    weightTable.heading("Weight exhaustive", text="Weight exhaustive", anchor=tk.W)
    weightTable.heading("Stock Name greedy", text="Stock Name greedy", anchor=tk.W)
    weightTable.heading("Weight greedy", text="Weight greedy", anchor=tk.W)
    weightTable.heading("Stock Name annealing", text="Stock Name annealing", anchor=tk.W)
    weightTable.heading("Weight annealing", text="Weight annealing", anchor=tk.W)

    names_exhaustive = dummyTrade.timeSeries.ticker[dummyTrade.portfolios.weights['exhaustive'][0].astype(bool)]
    weights_exhaustive = dummyTrade.portfolios.weights['exhaustive'][0][dummyTrade.portfolios.weights['exhaustive'][0] != 0]
    names_greedy = dummyTrade.timeSeries.ticker[dummyTrade.portfolios.weights['greedy'][0].astype(bool)]
    weights_greedy = dummyTrade.portfolios.xoptbuy0[dummyTrade.portfolios.xoptbuy0 != 0]
    names_annealing = dummyTrade.timeSeries.ticker[dummyTrade.portfolios.weights['annealing'][0].astype(bool)]
    weights_annealing = dummyTrade.portfolios.weights['annealing'][0][dummyTrade.portfolios.weights['annealing'][0] != 0]

    for count, data in enumerate(zip(names_exhaustive, weights_exhaustive, names_greedy, weights_greedy, names_annealing, weights_annealing)):
        weightTable.insert(parent='', index='end', iid=count, text="",
                           values=(data[0], data[1], data[2], data[3], data[4], data[5]))

    weightTable.pack()


def compute_returns(prices):
    """ Compute log returns of a time series """
    # return np.log(prices[1:] / prices[:-1])
    return prices[1:] - prices[:-1]
    # return prices


def rolling_return_distribution(window_size=250, bins=20):
    """ Compute and plot the distribution of returns using a sliding window """
    global dummyTrade
    figure9.clf()
    ax = figure9.add_subplot(111)

    returns = compute_returns(dummyTrade.portfolios.xoptbuy0.dot(dummyTrade.timeSeries.tr_Pperf))

    # Number of windows
    num_windows = len(returns) - window_size + 1

    # for i in range(num_windows):
    i = 0
    # Extract windowed returns
    windowed_returns = returns[i : i + window_size]

    # Plot histogram for the current window
    ax.hist(windowed_returns, bins=bins, alpha=0.3, density=True, label=f'Window {i+1}' if i == 0 else "")

    canvas9.draw()
    canvas9.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


def plot_2d_distribution(bins=50):
    """ Create a 2D distribution of price values and their corresponding returns """
    global dummyTrade
    figure9.clf()
    ax = figure9.add_subplot(111)
    returns = compute_returns(dummyTrade.portfolios.xoptbuy0.dot(dummyTrade.timeSeries.tr_Pperf))
    prices = dummyTrade.portfolios.xoptbuy0.dot(dummyTrade.timeSeries.tr_Pperf)[:-1]  # Align sizes (returns are one element shorter)

    # 2D Histogram (Density)
    # ax.hexbin(prices, returns, gridsize=bins, cmap='Blues')
    ax.hist(returns, bins=bins, edgecolor='black', alpha=0.7)
    # Scatter plot overlay (optional)
    # ax.scatter(prices, returns, alpha=0.3, color='red', s=5, label='Data Points')

    canvas9.draw()
    canvas9.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

filemenu.add_command(label='New', command=runTradings)
filemenu.add_command(label='Open', command=addApp)
filemenu.add_command(label='Save', command=runTradings)
filemenu.add_command(label='Close', command=runTradings)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=runTradings)

editmenu.add_command(label='Copy', command=runTradings)

helpmenu.add_command(label="Help Index", command=runTradings)

menubar.add_cascade(label='File', menu=filemenu)
menubar.add_cascade(label='Edit', menu=editmenu)
menubar.add_cascade(label='Help', menu=helpmenu)


global dummyTrade
dummyTrade = None

notebook = ttk.Notebook(root)
notebook.pack(pady=10, expand=True)

frame1 = ttk.Frame(notebook)
frame2 = ttk.Frame(notebook)
frame3 = ttk.Frame(notebook)
frame4 = ttk.Frame(notebook)
frame5 = ttk.Frame(notebook)
frame6 = ttk.Frame(notebook)
frame7 = ttk.Frame(notebook)
frame8 = ttk.Frame(notebook)

frame1.pack(fill=tk.BOTH, expand=True)
frame2.pack(fill=tk.BOTH, expand=True)
frame3.pack(fill=tk.BOTH, expand=True)
frame4.pack(fill=tk.BOTH, expand=True)
frame5.pack(fill=tk.BOTH, expand=True)
frame6.pack(fill=tk.BOTH, expand=True)
frame7.pack(fill=tk.BOTH, expand=True)
frame8.pack(fill=tk.BOTH, expand=True)

notebook.add(frame1, text='Trading')
notebook.add(frame2, text='Stocks')
notebook.add(frame3, text='VAR')
notebook.add(frame4, text='Seq2Seq train')
notebook.add(frame5, text='Hilbert curve')
notebook.add(frame6, text='Constituents')
notebook.add(frame7, text='Filter portfolio')
notebook.add(frame8, text='Return distributions')

lf1 = tk.LabelFrame(frame1, text="Trading")
lf2 = tk.LabelFrame(frame1, text="Trading")
lf3 = tk.LabelFrame(frame1, text="Weights")
lf4 = tk.LabelFrame(frame3, text="VAR fitting")
lf5 = tk.LabelFrame(frame4, text="Seq2Seq train results")
lf6 = tk.LabelFrame(frame5, text="Hilbert curve of portfolio")
lf7 = tk.LabelFrame(frame6, text="Stocks in the portfolio")
lf8 = tk.LabelFrame(frame7, text="Filter profile")
lf9 = tk.LabelFrame(frame8, text="Return distributions")

lf1.pack(side=tk.LEFT)
lf2.pack(side=tk.LEFT)
lf3.pack(side=tk.RIGHT)
lf4.pack(side=tk.LEFT)
lf5.pack(side=tk.LEFT)
lf6.pack(side=tk.LEFT)
lf7.pack(side=tk.LEFT)
lf8.pack(side=tk.LEFT)
lf9.pack(side=tk.LEFT)

inputText1 = tk.Entry(lf2, width=10)
inputText2 = tk.Entry(lf2, width=10)
inputText2.pack(side=tk.TOP, anchor=tk.NW)
inputText3 = tk.Entry(lf2, width=10)
inputText3.pack(side=tk.TOP, anchor=tk.NW)
inputText4 = tk.Entry(lf4, width=10)
inputText4.pack(side=tk.TOP, anchor=tk.NW)
inputText5 = tk.Entry(lf5, width=10)
inputText5.pack(side=tk.TOP, anchor=tk.NW)
ratio = tk.Entry(lf8, width=10)
ratio.pack(side=tk.TOP, anchor=tk.NW)

runTrading = tk.Button(lf2, text="Run Trading", padx=10, pady=5, fg="white", bg="#263D42", command=runTradings)
runTrading.pack(side=tk.TOP, anchor=tk.NW)

progress = tk.ttk.Progressbar(lf2, orient=tk.HORIZONTAL, length=150, mode='determinate')
progress.pack(side=tk.TOP, anchor=tk.NW)
# downloadStockData = tk.Button(lf1, text="Download stocks", padx=10, pady=5, fg="white", bg="#263D42", command=getStocks)
# downloadStockData.pack()

saveImage = tk.Button(lf2, text="Save image", padx=10, pady=5, fg="white", bg="#263D42", command=saveImage)
saveImage.pack(side=tk.TOP, anchor=tk.NW)

plotStockData = tk.Button(frame2, text="Plot stocks", padx=10, pady=5, fg="white", bg="#263D42", command=plotStockPrices)
plotStockData.pack()

saveDummyTrade = tk.Button(lf2, text="Save Trade", padx=10, pady=5, fg="white", bg="#263D42", command=saveTrade)
saveDummyTrade.pack(side=tk.TOP, anchor=tk.N)

loadDummyTrade = tk.Button(lf2, text="Load Trade", padx=10, pady=5, fg="white", bg="#263D42", command=loadTrade)
loadDummyTrade.pack(side=tk.TOP, anchor=tk.N)

fittVarP = tk.Button(lf4, text="Fit VAR(p)", padx=10, pady=5, fg="white", bg="#263D42", command=calibrateVAR)
fittVarP.pack(side=tk.TOP, anchor=tk.N)

replotVarP = tk.Button(lf4, text="Replot VAR(p)", padx=10, pady=5, fg="white", bg="#263D42", command=replotVAR)
replotVarP.pack(side=tk.TOP, anchor=tk.N)

plotVarP = tk.Button(lf4, text="Compare p's of VAR(p)", padx=10, pady=5, fg="white", bg="#263D42", command=plotVAR)
plotVarP.pack(side=tk.TOP, anchor=tk.N)

plotSeq2Seq = tk.Button(lf5, text="Plot Seq2Seq train result", padx=10, pady=5, fg="white", bg="#263D42", command=plotSeq2SeqTrainResult)
plotSeq2Seq.pack(side=tk.TOP, anchor=tk.N)
plotSeq2Seq = tk.Button(lf5, text="Up", padx=10, pady=5, fg="white", bg="#263D42", command=upPlotSeq2SeqTrainResult)
plotSeq2Seq.pack(side=tk.TOP, anchor=tk.N)
plotSeq2Seq = tk.Button(lf5, text="Down", padx=10, pady=5, fg="white", bg="#263D42", command=downPlotSeq2SeqTrainResult)
plotSeq2Seq.pack(side=tk.TOP, anchor=tk.N)

plotFilter = tk.Button(lf8, text="Replot FFT", padx=10, pady=5, fg="white", bg="#263D42", command=replotFFT)
plotFilter.pack(side=tk.TOP, anchor=tk.N)

str1 = tk.StringVar(frame1)
str2 = tk.StringVar(frame1)
str3 = tk.StringVar(frame1)
str4 = tk.StringVar(frame1)
str5 = tk.StringVar(frame1)
str6 = tk.StringVar(frame1)
var1 = tk.IntVar(frame1)
var2 = tk.IntVar(frame1)
var3 = tk.IntVar(frame1)
var4 = tk.IntVar(frame3)

# Dictionary to create multiple buttons
values = {"RadioButton 1": "1",
          "RadioButton 2": "2",
          "RadioButton 3": "3",
          "RadioButton 4": "4",
          "RadioButton 5": "5"}

# Loop is used to create multiple Radiobuttons
# rather than creating each button separately

c1 = tk.Checkbutton(lf2, text='VAR1', variable=str1, onvalue="VAR1", offvalue="", command=plotPortfolioAndPrediction)
c1.pack(side=tk.TOP)
c2 = tk.Checkbutton(lf2, text='Trading Range', variable=str2, onvalue="Ranges", offvalue="", command=plotPortfolioAndPrediction)
c2.pack(side=tk.TOP)
c3 = tk.Checkbutton(lf2, text='Online LSTM', variable=str3, onvalue="Online LSTM", offvalue="", command=plotPortfolioAndPrediction)
c3.pack(side=tk.TOP)
c4 = tk.Checkbutton(lf2, text='Offline LSTM', variable=str4, onvalue="Offline LSTM", offvalue="", command=plotPortfolioAndPrediction)
c4.pack(side=tk.TOP)
c5 = tk.Checkbutton(lf2, text='Seq2Seq LSTM', variable=str5, onvalue="Seq2Seq LSTM", offvalue="", command=plotPortfolioAndPrediction)
c5.pack(side=tk.TOP)
c6 = tk.Checkbutton(lf2, text='NLP thread LSTM', variable=str6, onvalue="NLP thread LSTM", offvalue="", command=plotPortfolioAndPrediction)
c6.pack(side=tk.TOP)


selected = tk.StringVar(frame2)
options = [str(i) for i in range(400)]
drop = tk.OptionMenu(frame2, selected, *options)
drop.pack()

selectedFilter = tk.StringVar(frame7)
possibleFilters = ["blackman", "hamming", "gaussian", "triangular", "tukey", "plato"]
dropFilter = tk.OptionMenu(lf8, selectedFilter, *possibleFilters)
dropFilter.pack()


figure1 = plt.Figure(figsize=(8, 8), dpi=100)
ax1 = figure1.add_subplot(111)
bar1 = FigureCanvasTkAgg(figure1, lf2)
bar1.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH)

figure2 = plt.Figure(figsize=(8, 10), dpi=100)
ax2 = figure2.add_subplot(111)
canvas2 = FigureCanvasTkAgg(figure2, frame2)
canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH)
#
figure3 = plt.Figure(figsize=(8, 4), dpi=100)
ax3 = figure3.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas3 = FigureCanvasTkAgg(figure3, root)
canvas3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


figure4 = plt.Figure(figsize=(8, 8), dpi=100)
ax4 = figure4.add_subplot(111)
canvas4 = FigureCanvasTkAgg(figure4, lf4)
canvas4.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

figure5 = plt.Figure(figsize=(8, 8), dpi=100)
ax5 = figure5.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas5 = FigureCanvasTkAgg(figure5, lf5)
canvas5.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

figure6 = plt.Figure(figsize=(8, 8), dpi=100)
ax6 = figure6.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas6 = FigureCanvasTkAgg(figure6, lf6)
canvas6.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

figure7 = plt.Figure(figsize=(8, 8), dpi=100)
ax7 = figure7.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas7 = FigureCanvasTkAgg(figure7, lf7)
canvas7.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

figure8 = plt.Figure(figsize=(8, 8), dpi=100)
ax8 = figure8.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas8 = FigureCanvasTkAgg(figure8, lf8)
canvas8.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

figure9 = plt.Figure(figsize=(8, 8), dpi=100)
ax9 = figure9.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas9 = FigureCanvasTkAgg(figure9, lf9)
canvas9.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


root.config(menu=menubar)
root.mainloop()

with open("save.txt", "w") as f:
    for app in apps:
        f.write(app + ',')