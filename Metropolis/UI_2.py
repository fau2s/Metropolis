import pickle
import tkinter as tk
import random
from tkinter import filedialog, Text, ttk
import os

import LuongAttention
import Trading_2, Trading, getStockData
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import threading
import matplotlib.animation as animation

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


def update_progress(value):
    progress["value"] = value
    lf2.update_idletasks()


def runTradings():
    global dummyTrade
    dummyTrade = Trading.Trading(
        n=int(nInputText.get()),
        spars=int(sparInputText.get()),
        simulated=False,
        idTimeWindow=int(inputText1.get()), #400
        tradingStart=int(inputText2.get()), #1900
        tradingTimeWindow=int(inputText3.get()), #150
        useShift=True,
        n_diff=int(nDiffInputText.get()),
        model=selected_option.get(),
        sigma=1e-1,
        progress_callback = update_progress
    )
    # dummyTrade.calibrating()
    thread = threading.Thread(target=dummyTrade.calibrating)
    thread.start()
    # dummyTrade.process()
    # trade.set(dummyTrade)
    dummyTrade.plotSimplePortfolio(figure1)
    bar1.draw()
    bar1.get_tk_widget().grid(row=8, column=0, sticky="nsew")
    dummyTrade.plotSeq2SeqTrainingResult(figure5)
    canvas5.draw()
    canvas5.get_tk_widget().grid(row=1, column=0, sticky="nsew")
    dummyTrade.plotConstituents(figure7)
    canvas7.draw()
    canvas7.get_tk_widget().grid(row=1, column=0, sticky="nsew")
    dummyTrade.plotFilter(figure8)
    canvas8.draw()
    canvas8.get_tk_widget().grid(row=1, column=0, sticky="nsew")
    # rolling_return_distribution(window_size=350)
    plot_2d_distribution(bins=30)
    canvas9.draw()
    canvas9.get_tk_widget().grid(row=1, column=0, sticky="nsew")

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
    bar1.get_tk_widget().grid(row=1, column=0, sticky="nsew")


def plotOfflineLSTM():
    global dummyTrade
    if dummyTrade is None:
        return
    if var3.get():
        dummyTrade.plotPortfolio(dummyTrade.nlpPred, figure1)
    if not var3.get():
        dummyTrade.plotSimplePortfolio(figure1)

    bar1.draw()
    bar1.get_tk_widget().grid(row=1, column=0, sticky="nsew")


def plotTradingRange():
    global dummyTrade
    if dummyTrade is None:
        return
    if var2.get():
        dummyTrade.plotTest(dummyTrade.nlpPred, figure1)
    if not var2.get():
        dummyTrade.plotSimplePortfolio(figure1)

    bar1.draw()
    bar1.get_tk_widget().grid(row=1, column=0, sticky="nsew")


def plotPortfolioAndPrediction():
    global dummyTrade
    if dummyTrade is None:
        return
    option = [str1.get(), str2.get(), str3.get(), str4.get(), str5.get(), str6.get()]
    dummyTrade.plotPortfolioAndPredition(dummyTrade.nlpPred, figure1, option)
    bar1.draw()
    bar1.get_tk_widget().grid(row=1, column=0, sticky="nsew")

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
    canvas2.get_tk_widget().grid(row=2, column=0, padx=10, pady=10, sticky="nw")


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
                    n=int(nInputText.get()),
                    spars=int(sparInputText.get()),
                    simulated=False,
                    idTimeWindow=int(inputText1.get()),  # 400
                    tradingStart=int(inputText2.get()),  # 1900
                    tradingTimeWindow=int(inputText3.get()),  # 150
                    useShift=True,
                    n_diff=int(nDiffInputText.get()),
                    model=selected_option.get(),
                    sigma=1e-1,
                )
                dummyTrade.timeSeries.estA()
                dummyTrade.timeSeries.estAVarP(varp) #dummyTrade.timeSeries.estAVarP(int(inputText4.get()))
                randIndex = randint(0, int(nInputText.get()) - 1)
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
    randIndex = randint(0, int(nInputText.get()) - 1)
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
    canvas4.get_tk_widget().grid(row=1, column=0, sticky="nsew")

def plotVAR():
    global dummyTrade
    dummyTrade = Trading.Trading(
        n=int(nInputText.get()),
        spars=int(sparInputText.get()),
        simulated=False,
        idTimeWindow=int(inputText1.get()), #400
        tradingStart=int(inputText2.get()), #1900
        tradingTimeWindow=int(inputText3.get()), #150
        useShift=True,
        n_diff=int(nDiffInputText.get()),
        model=selected_option.get(),
        sigma=1e-1,
    )
    figure4.clf()
    ax1 = figure4.add_subplot(111)
    randIndex = randint(0, int(nInputText.get()) - 1)
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
    canvas4.get_tk_widget().grid(row=1, column=0, sticky="nsew")


def plotSeq2SeqTrainResult():
    global dummyTrade
    dummyTrade.plotSeq2SeqTrainingResult(figure5, int(inputText5.get()))
    canvas5.draw()
    canvas5.get_tk_widget().grid(row=2, column=0, sticky="nsew")


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
    canvas5.get_tk_widget().grid(row=1, column=0, sticky="nsew")


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
    canvas5.get_tk_widget().grid(row=2, column=0, sticky="nsew")
# canvas = tk.Canvas(root, height=900, width=1800, bg="#263D42")
# canvas.create_text(100, 100, text=getStocks, fill="black")
# canvas.pack()

def plotConstituents():
    global dummyTrade

def replotFFT():
    global dummyTrade
    dummyTrade.plotFilter(figure8, selectedFilter.get(), float(ratio.get()))
    canvas8.draw()
    canvas8.get_tk_widget().grid(row=1, column=0, sticky="nsew")


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


def createPortfolio():
    global dummyTrade
    dummyTrade = Trading.Trading(
        n=int(nInputText.get()),
        spars=int(sparInputText.get()),
        simulated=False,
        idTimeWindow=int(inputText1.get()), #400
        tradingStart=int(inputText2.get()), #1900
        tradingTimeWindow=int(inputText3.get()), #150
        useShift=True,
        n_diff=int(nDiffInputText.get()),
        model=selected_option.get(),
        sigma=1e-1,
        progress_callback=update_progress
    )


def runCalib():
    global dummyTrade
    tradingdata = np.concatenate(
        (dummyTrade.idPortfolio(),  # self.portfolios.xoptbuy0.dot(self.timeSeries.Pid),
         dummyTrade.perfPortfolio())
    )
    nlpLSTMPred = LuongAttention.NLPwithLSTM(
        data=tradingdata,
        train_ratio=dummyTrade.trainRatio(),
        n_hidden=int(numOfHL_inputText.get()),
        momentum=float(momentum_inputText.get()),
        epc=int(epoch_input.get()),
        learning_rate=float(lr_input.get()),
        batch_size=int(batch_input.get()),
        train_len=int(truncLength_input.get()),
        test_len=int(predLength_input.get()),
        attention="Luong"
    )
    nlpLSTMPred.constructnetwork()
    thread = threading.Thread(target=nlpLSTMPred.trainnetwork())
    thread.start()
    train_pred_detrend = nlpLSTMPred.model.predict(nlpLSTMPred.X_input_train[:, :, :nlpLSTMPred.numberOfFeatures + 1]) * nlpLSTMPred.x_train_max[0]
    test_pred_detrend = nlpLSTMPred.model.predict(nlpLSTMPred.X_input_test[:, :, :nlpLSTMPred.numberOfFeatures + 1]) * nlpLSTMPred.x_train_max[0]
    train_true_detrend = nlpLSTMPred.X_output_train[:, :, :nlpLSTMPred.numberOfFeatures + 1] * nlpLSTMPred.x_train_max[0]
    test_true_detrend = nlpLSTMPred.X_output_test[:, :, :nlpLSTMPred.numberOfFeatures + 1] * nlpLSTMPred.x_train_max[0]
    data_final = dict()
    for dt, lb in zip([train_pred_detrend, train_true_detrend, test_pred_detrend, test_true_detrend],
                      ['train_pred', 'train_true', 'test_pred', 'test_true']):
        dt_x1 = dt[:, :, 0]
        dt_x2 = dt[:, :, 1] / nlpLSTMPred.x_train_max[0]
        data_final[lb] = np.concatenate(
            [np.expand_dims(dt_x1, axis=2), np.expand_dims(dt_x2, axis=2)], axis=2) // np.expand_dims(dt_x1, axis=2)

    ith_timestep = random.choice(range(data_final['train_pred'].shape[1]))
    train_start_t = 0
    test_start_t = data_final['train_pred'].shape[0]
    x_lbl = 'x1'
    figure10.clf()
    ax10 = figure10.add_subplot(111)
    ax10.plot(range(0, 0 + data_final['train_pred'].shape[0]),
             data_final['train_pred'][:, ith_timestep, 0],
             linestyle='--', linewidth=1, color='green', label='pred ' + x_lbl)
    ax10.plot(range(0, 0 + data_final['train_pred'].shape[0]),
             data_final['train_true'][:, ith_timestep, 0],
             linestyle='-', linewidth=1, color='blue', label='true ' + x_lbl)
    ax10.plot(range(test_start_t, test_start_t + data_final['test_pred'].shape[0]),
             data_final['test_pred'][:, ith_timestep, 0],
             linestyle='--', linewidth=1, color='red', label='pred ' + x_lbl)
    ax10.plot(range(test_start_t, test_start_t + data_final['test_pred'].shape[0]),
             data_final['test_true'][:, ith_timestep, 0],
             linestyle='-', linewidth=1, color='orange', label='true ' + x_lbl)

    canvas10.draw()
    canvas10.get_tk_widget().grid(row=1, column=0, sticky="nsew")

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
frame9 = ttk.Frame(notebook)
frame10 = ttk.Frame(notebook)

frame1.pack(fill=tk.BOTH, expand=True)
frame2.pack(fill=tk.BOTH, expand=True)
frame3.pack(fill=tk.BOTH, expand=True)
frame4.pack(fill=tk.BOTH, expand=True)
frame5.pack(fill=tk.BOTH, expand=True)
frame6.pack(fill=tk.BOTH, expand=True)
frame7.pack(fill=tk.BOTH, expand=True)
frame8.pack(fill=tk.BOTH, expand=True)
frame9.pack(fill=tk.BOTH, expand=True)
frame10.pack(fill=tk.BOTH, expand=True)

notebook.add(frame1, text='Trading')
notebook.add(frame2, text='Stocks')
notebook.add(frame3, text='VAR')
notebook.add(frame4, text='Seq2Seq train')
notebook.add(frame5, text='Hilbert curve')
notebook.add(frame6, text='Constituents')
notebook.add(frame7, text='Filter portfolio')
notebook.add(frame8, text='Return distributions')
notebook.add(frame9, text='VAR regression matrix')
notebook.add(frame10, text='Seq2Seq LSTM calibration')

lf1 = tk.LabelFrame(frame1, text="Trading")
lf2 = tk.LabelFrame(frame1, text="Trading")
lf3 = tk.LabelFrame(frame1, text="Weights")
lf4 = tk.LabelFrame(frame3, text="VAR fitting")
lf5 = tk.LabelFrame(frame4, text="Seq2Seq train results")
lf6 = tk.LabelFrame(frame5, text="Hilbert curve of portfolio")
lf7 = tk.LabelFrame(frame6, text="Stocks in the portfolio")
lf8 = tk.LabelFrame(frame7, text="Filter profile")
lf9 = tk.LabelFrame(frame8, text="Return distributions")
lf10 = tk.LabelFrame(frame9, text="VAR regression matrix")
lf11 = tk.LabelFrame(frame10, text="Seq2Seq LSTM calibration")

lf1.pack(side=tk.LEFT)
lf2.pack(side=tk.LEFT)
lf3.pack(side=tk.RIGHT)
lf4.pack(side=tk.LEFT)
lf5.pack(side=tk.LEFT)
lf6.pack(side=tk.LEFT)
lf7.pack(side=tk.LEFT)
lf8.pack(side=tk.LEFT)
lf9.pack(side=tk.LEFT)
lf10.pack(side=tk.LEFT)
lf11.pack(side=tk.LEFT)

lf_trading = ttk.LabelFrame(lf2, text="Trading Controls")
lf_trading.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

label1 = tk.Label(lf_trading, text="Calibration Length")
label1.grid(row=0, column=0, sticky="w", padx=5, pady=2)
inputText1 = tk.Entry(lf_trading, width=10)
inputText1.insert(0, 500)
inputText1.grid(row=0, column=1, sticky="w", padx=5, pady=2)
label2 = tk.Label(lf_trading, text="Start of Trading")
label2.grid(row=1, column=0, sticky="w", padx=5, pady=2)
inputText2 = tk.Entry(lf_trading, width=10)
inputText2.insert(0, 1789)
inputText2.grid(row=1, column=1, sticky="w", padx=5, pady=2)
label3 = tk.Label(lf_trading, text="Length of Trading")
label3.grid(row=2, column=0, sticky="w", padx=5, pady=2)
inputText3 = tk.Entry(lf_trading, width=10)
inputText3.insert(0, 150)
inputText3.grid(row=2, column=1, sticky="w", padx=5, pady=2)
label4 = tk.Label(lf_trading, text="Number of Stocks")
label4.grid(row=3, column=0, sticky="w", padx=5, pady=2)
nInputText = tk.Entry(lf_trading, width=10)
nInputText.insert(0, 430)
nInputText.grid(row=3, column=1, sticky="w", padx=5, pady=2)
label5 = tk.Label(lf_trading, text="Sparsity")
label5.grid(row=4, column=0, sticky="w", padx=5, pady=2)
sparInputText = tk.Entry(lf_trading, width=10)
sparInputText.insert(0, 8)
sparInputText.grid(row=4, column=1, sticky="w", padx=5, pady=2)
label6 = tk.Label(lf_trading, text="Number of Differentiations")
label6.grid(row=5, column=0, sticky="w", padx=5, pady=2)
nDiffInputText = tk.Entry(lf_trading, width=10)
nDiffInputText.insert(0, 0)
nDiffInputText.grid(row=5, column=1, sticky="w", padx=5, pady=2)
label7 = tk.Label(lf_trading, text="Data Model")
label7.grid(row=6, column=0, sticky="w", padx=5, pady=2)
options = ['regular', 'diff', 'log', 'logdiff']
selected_option = tk.StringVar(value=options[0])  # Default value
dropdown = ttk.Combobox(lf_trading, width=7, textvariable=selected_option, values=options, state='readonly')
dropdown.grid(row=6, column=1, sticky="w", padx=5, pady=2)

# Function to print the selected option
def on_select(event):
    print("Selected option:", selected_option.get())

# Bind the dropdown selection to the function
dropdown.bind("<<ComboboxSelected>>", on_select)

inputText4 = tk.Entry(lf4, width=10)
inputText4.grid(row=0, column=0, padx=5, pady=2)
inputText5 = tk.Entry(lf5, width=10)
inputText5.grid(row=0, column=0, padx=5, pady=2)
ratio = tk.Entry(lf8, width=10)
ratio.grid(row=0, column=0, padx=5, pady=2)

runTrading = tk.Button(lf_trading, text="Run Trading", padx=10, pady=5, fg="white", bg="#263D42", command=runTradings)
runTrading.grid(row=0, column=2, padx=10, pady=5)

progress = tk.ttk.Progressbar(lf_trading, orient=tk.HORIZONTAL, length=150, mode='determinate')
progress.grid(row=7, column=2, columnspan=3, pady=5)
# downloadStockData = tk.Button(lf1, text="Download stocks", padx=10, pady=5, fg="white", bg="#263D42", command=getStocks)
# downloadStockData.pack()

saveImage = tk.Button(lf_trading, text="Save image", padx=10, pady=5, fg="white", bg="#263D42", command=saveImage)
saveImage.grid(row=1, column=2, padx=10, pady=5)

plotStockData = tk.Button(frame2, text="Plot stocks", padx=10, pady=5, fg="white", bg="#263D42", command=plotStockPrices)
plotStockData.grid(row=0, column=0, padx=10, pady=5)

saveDummyTrade = tk.Button(lf_trading, text="Save Trade", padx=10, pady=5, fg="white", bg="#263D42", command=saveTrade)
saveDummyTrade.grid(row=2, column=2, padx=10, pady=5)

loadDummyTrade = tk.Button(lf_trading, text="Load Trade", padx=10, pady=5, fg="white", bg="#263D42", command=loadTrade)
loadDummyTrade.grid(row=3, column=2, padx=10, pady=5)

fittVarP = tk.Button(lf4, text="Fit VAR(p)", padx=10, pady=5, fg="white", bg="#263D42", command=calibrateVAR)
fittVarP.grid(row=0, column=1, padx=10, pady=5)

replotVarP = tk.Button(lf4, text="Replot VAR(p)", padx=10, pady=5, fg="white", bg="#263D42", command=replotVAR)
replotVarP.grid(row=0, column=2, padx=10, pady=5)

plotVarP = tk.Button(lf4, text="Compare p's of VAR(p)", padx=10, pady=5, fg="white", bg="#263D42", command=plotVAR)
plotVarP.grid(row=0, column=3, padx=10, pady=5)

plotSeq2Seq = tk.Button(lf5, text="Plot Seq2Seq train result", padx=10, pady=5, fg="white", bg="#263D42", command=plotSeq2SeqTrainResult)
plotSeq2Seq.grid(row=0, column=1, padx=10, pady=5)
plotSeq2Seq = tk.Button(lf5, text="Up", padx=10, pady=5, fg="white", bg="#263D42", command=upPlotSeq2SeqTrainResult)
plotSeq2Seq.grid(row=0, column=2, padx=10, pady=5)
plotSeq2Seq = tk.Button(lf5, text="Down", padx=10, pady=5, fg="white", bg="#263D42", command=downPlotSeq2SeqTrainResult)
plotSeq2Seq.grid(row=1, column=2, padx=10, pady=5)

plotFilter = tk.Button(lf8, text="Replot FFT", padx=10, pady=5, fg="white", bg="#263D42", command=replotFFT)
plotFilter.grid(row=0, column=1, padx=10, pady=5)

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

c1 = tk.Checkbutton(lf_trading, text='VAR1', variable=str1, onvalue="VAR1", offvalue="", command=plotPortfolioAndPrediction)
c1.grid(row=0, column=3, columnspan=3, pady=10, sticky="ew")
c2 = tk.Checkbutton(lf_trading, text='Trading Range', variable=str2, onvalue="Ranges", offvalue="", command=plotPortfolioAndPrediction)
c2.grid(row=1, column=3, columnspan=3, pady=10, sticky="ew")
c3 = tk.Checkbutton(lf_trading, text='Online LSTM', variable=str3, onvalue="Online LSTM", offvalue="", command=plotPortfolioAndPrediction)
c3.grid(row=2, column=3, columnspan=3, pady=10, sticky="ew")
c4 = tk.Checkbutton(lf_trading, text='Offline LSTM', variable=str4, onvalue="Offline LSTM", offvalue="", command=plotPortfolioAndPrediction)
c4.grid(row=3, column=3, columnspan=3, pady=10, sticky="ew")
c5 = tk.Checkbutton(lf_trading, text='Seq2Seq LSTM', variable=str5, onvalue="Seq2Seq LSTM", offvalue="", command=plotPortfolioAndPrediction)
c5.grid(row=4, column=3, columnspan=3, pady=10, sticky="ew")
c6 = tk.Checkbutton(lf_trading, text='NLP thread LSTM', variable=str6, onvalue="NLP thread LSTM", offvalue="", command=plotPortfolioAndPrediction)
c6.grid(row=5, column=3, columnspan=3, pady=10, sticky="ew")


selected = tk.StringVar(frame2)
options = [str(i) for i in range(400)]
drop = tk.OptionMenu(frame2, selected, *options)
drop.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

selectedFilter = tk.StringVar(frame7)
possibleFilters = ["blackman", "hamming", "gaussian", "triangular", "tukey", "plato"]
dropFilter = tk.OptionMenu(lf8, selectedFilter, *possibleFilters)
dropFilter.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

input_frame = ttk.LabelFrame(lf10, text="Inputs")
input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

ttk.Label(input_frame, text="Calibration Length (l):").grid(row=0, column=0, padx=5, pady=5)
calib_length_entry = ttk.Entry(input_frame, width=10)
calib_length_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(input_frame, text="Calibration Start (s):").grid(row=1, column=0, padx=5, pady=5)
calib_start_entry = ttk.Entry(input_frame, width=10)
calib_start_entry.grid(row=1, column=1, padx=5, pady=5)


# Plot frame
plot_frame = ttk.LabelFrame(lf10, text="Regression Matrix Video")
plot_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

figure, ax = plt.subplots(figsize=(5, 5))
canvas = FigureCanvasTkAgg(figure, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def runVARCalibration():
    try:
        l = int(calib_length_entry.get())
        s = int(calib_start_entry.get())
    except ValueError:
        print("Please enter valid integers for calibration length and start.")
        return

    def update_frame(i):
        figure.clf()  # Clear the previous plot
        # Draw current frame (regression matrix figure)
        frame_fig = frames[i]
        # Transfer the frame's Axes image to the main figure
        ax_main = figure.add_subplot(111)
        ax_frame = frame_fig.axes[0]

        # Transfer matrix image
        im = ax_frame.images[0]
        ax_main.imshow(im.get_array(), cmap=im.get_cmap(), vmin=-0.05, vmax=.05)
        ax_main.set_title(ax_frame.get_title())

        # Transfer colorbar
        figure.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)

    # if l <= 0 or s < 0 or s + l >= L:
    #     print("Invalid l or s values. Ensure 0 <= s and s + l < L.")
    #     return

    # Calibration video frames
    frames = []
    dummyTrade = Trading.Trading(
        n=int(nInputText.get()),
        spars=int(sparInputText.get()),
        simulated=False,
        idTimeWindow=l,  # 400
        tradingStart=s,  # 1900
        tradingTimeWindow=int(inputText3.get()),  # 150
        useShift=True,
        n_diff=0,
        model=selected_option.get(),
        sigma=1e-1,
    )
    for t in range(s, s+30):
        # window_data = dummyTrade.timeSeries[:, t:t + l].T  # shape (l, N)
        # df_window = pd.DataFrame(window_data)
        dummyTrade = Trading.Trading(
            n=int(nInputText.get()),
            spars=int(sparInputText.get()),
            simulated=False,
            idTimeWindow=l,  # 400
            tradingStart=t,  # 1900
            tradingTimeWindow=int(inputText3.get()),  # 150
            useShift=True,
            n_diff=0,
            model=selected_option.get(),
            sigma=1e-1,
        )
        # dummyTrade.timeSeries.estA()
        results = dummyTrade.timeSeries.estAmatrixVarP(2, 0)  # dummyTrade.timeSeries.estAVarP(int(inputText4.get()))
        reg_matrix = results  # shape (N, N)

        # Plotting regression matrix
        fig, ax = plt.subplots(figsize=(4, 4))
        cax = ax.matshow(reg_matrix, cmap='viridis', )
        ax.set_title(f"Time {t}")
        plt.colorbar(cax, ax=ax)
        plt.close(fig)  # Close to avoid memory leaks

        frames.append(fig)

    # Start animation
    ani = animation.FuncAnimation(figure, update_frame, frames=len(frames), interval=500,
                                       repeat=True)
    canvas.draw()




ttk.Button(input_frame, text="Run VAR Calibration", command=runVARCalibration).grid(row=2, column=0, columnspan=2,
                                                                                   pady=10)


lstm_calib = ttk.LabelFrame(lf11, text="Seq2Seq LSTM calibration")
lstm_calib.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

numOFHL_label = tk.Label(lstm_calib, text="Number of hidden layers")
numOFHL_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
numOfHL_inputText = tk.Entry(lstm_calib, width=10)
numOfHL_inputText.insert(0, 64)
numOfHL_inputText.grid(row=0, column=1, sticky="w", padx=5, pady=2)
momentum_label = tk.Label(lstm_calib, text="Momentum")
momentum_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
momentum_inputText = tk.Entry(lstm_calib, width=10)
momentum_inputText.insert(0, 0.5)
momentum_inputText.grid(row=1, column=1, sticky="w", padx=5, pady=2)
epoch_label = tk.Label(lstm_calib, text="Epoch")
epoch_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)
epoch_input = tk.Entry(lstm_calib, width=10)
epoch_input.insert(0, 500)
epoch_input.grid(row=2, column=1, sticky="w", padx=5, pady=2)
lr_label = tk.Label(lstm_calib, text="Learning Rate")
lr_label.grid(row=3, column=0, sticky="w", padx=5, pady=2)
lr_input = tk.Entry(lstm_calib, width=10)
lr_input.insert(0, 1)
lr_input.grid(row=3, column=1, sticky="w", padx=5, pady=2)
batch_label = tk.Label(lstm_calib, text="Batch Size")
batch_label.grid(row=4, column=0, sticky="w", padx=5, pady=2)
batch_input = tk.Entry(lstm_calib, width=10)
batch_input.insert(0, 100)
batch_input.grid(row=4, column=1, sticky="w", padx=5, pady=2)
trainRatio_label = tk.Label(lstm_calib, text="Train Ratio")
trainRatio_label.grid(row=5, column=0, sticky="w", padx=5, pady=2)
trainRatio_input = tk.Entry(lstm_calib, width=10)
trainRatio_input.insert(0, 0.2)
trainRatio_input.grid(row=5, column=1, sticky="w", padx=5, pady=2)
truncLength_label = tk.Label(lstm_calib, text="Truncation Length")
truncLength_label.grid(row=6, column=0, sticky="w", padx=5, pady=2)
truncLength_input = tk.Entry(lstm_calib, width=10)
truncLength_input.insert(0, 500)
truncLength_input.grid(row=6, column=1, sticky="w", padx=5, pady=2)
predLength_label = tk.Label(lstm_calib, text="Prediction Length")
predLength_label.grid(row=7, column=0, sticky="w", padx=5, pady=2)
predLength_input = tk.Entry(lstm_calib, width=10)
predLength_input.insert(0, 20)
predLength_input.grid(row=7, column=1, sticky="w", padx=5, pady=2)
# label7 = tk.Label(lstm_calib, text="Data Model")
# label7.grid(row=6, column=0, sticky="w", padx=5, pady=2)
# options = ['regular', 'diff', 'log', 'logdiff']
# selected_option = tk.StringVar(value=options[0])  # Default value
# dropdown = ttk.Combobox(lstm_calib, width=7, textvariable=selected_option, values=options, state='readonly')
# dropdown.grid(row=6, column=1, sticky="w", padx=5, pady=2)
createPortfolio = tk.Button(lstm_calib, text="Create Portfolio", padx=10, pady=5, fg="white", bg="#263D42", command=createPortfolio)
createPortfolio.grid(row=0, column=2, padx=10, pady=5)

runCalib = tk.Button(lstm_calib, text="Run Calibration", padx=10, pady=5, fg="white", bg="#263D42", command=runCalib)
runCalib.grid(row=1, column=2, padx=10, pady=5)




figure1 = plt.Figure(figsize=(8, 8), dpi=100)
ax1 = figure1.add_subplot(111)
bar1 = FigureCanvasTkAgg(figure1, lf2)
bar1.get_tk_widget().grid(row=8, column=0, sticky="nsew")

figure2 = plt.Figure(figsize=(8, 10), dpi=100)
ax2 = figure2.add_subplot(111)
canvas2 = FigureCanvasTkAgg(figure2, frame2)
canvas2.get_tk_widget().grid(row=2, column=0, sticky="nsew")
#
figure3 = plt.Figure(figsize=(8, 4), dpi=100)
ax3 = figure3.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas3 = FigureCanvasTkAgg(figure3, root)
canvas3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


figure4 = plt.Figure(figsize=(8, 8), dpi=100)
ax4 = figure4.add_subplot(111)
canvas4 = FigureCanvasTkAgg(figure4, lf4)
canvas4.get_tk_widget().grid(row=1, column=0, sticky="nsew")

figure5 = plt.Figure(figsize=(8, 8), dpi=100)
ax5 = figure5.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas5 = FigureCanvasTkAgg(figure5, lf5)
canvas5.get_tk_widget().grid(row=2, column=0, sticky="nsew")

figure6 = plt.Figure(figsize=(8, 8), dpi=100)
ax6 = figure6.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas6 = FigureCanvasTkAgg(figure6, lf6)
canvas6.get_tk_widget().grid(row=1, column=0, sticky="nsew")

figure7 = plt.Figure(figsize=(8, 8), dpi=100)
ax7 = figure7.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas7 = FigureCanvasTkAgg(figure7, lf7)
canvas7.get_tk_widget().grid(row=1, column=0, sticky="nsew")

figure8 = plt.Figure(figsize=(8, 8), dpi=100)
ax8 = figure8.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas8 = FigureCanvasTkAgg(figure8, lf8)
canvas8.get_tk_widget().grid(row=1, column=0, sticky="nsew")

figure9 = plt.Figure(figsize=(8, 8), dpi=100)
ax9 = figure9.add_subplot(111)
# ax3.scatter(df3['interest_rate'], df3['index_price'], color='g')
canvas9 = FigureCanvasTkAgg(figure9, lf9)
canvas9.get_tk_widget().grid(row=1, column=0, sticky="nsew")

figure10 = plt.Figure(figsize=(8, 8), dpi=100)
ax10 = figure10.add_subplot(111)
canvas10 = FigureCanvasTkAgg(figure10, lf11)
canvas10.get_tk_widget().grid(row=1, column=0, sticky="nsew")

root.config(menu=menubar)
root.mainloop()

with open("save.txt", "w") as f:
    for app in apps:
        f.write(app + ',')