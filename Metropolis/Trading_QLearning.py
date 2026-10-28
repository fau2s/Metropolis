import tensorflow as tf
import numpy as np
import pandas as pd

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# import numpy as np
# from binanceEnv import CryptoTradingEnv
import random
# changes the matplotlib's figure to black (optional)
style.use('dark_background')

# from Visualization import TradingGraph as tg

tf.compat.v1.enable_v2_behavior()


class CryptoTradingEnv(py_environment.PyEnvironment):
    def __init__(self, fee=0.001, starting_date="2018-05-29", initial_balance=10000, look_back_window=40):
        '''initialize the action spec and observation spec,
           the two main components of agent will need to navigate the environment'''
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.int32, minimum=0, maximum=10, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4, 5, 40), dtype=np.float32, minimum=0, name='observation')
        self._state = np.zeros((4, 5, 40), dtype=np.float32)

        # parameters specific to the environment
        self._episode_ended = False
        self.initial_balance = initial_balance
        self.wallet = [self.initial_balance]
        self.look_back_window = look_back_window  # how far back our agent will observe to make it's prediction
        self.fee = fee  # binance's commission fee per trade is 0.001
        self.starting_date = starting_date

        self.visual = None  # used for the rendering and visualization

        # set the initial step to the look back window so that our agent will have data to view on it's first step
        self.current_step = self.look_back_window

        self.moves = []  # stores the trades made by our agent

        ef = pd.ExcelFile("priceData.xls")  # used to get the sheets in our excel file
        self.dfs = []

        '''
        Iterate over the sheets (pairs/markets) and:
            - append an empty list for each coin to store the trades,
            - initialize each coin in our wallet to 0.0
            - load each sheet as a dataframe and store each df in the list self.dfs
        '''
        for sn in ef.sheet_names:
            print("loading sheet: " + sn)
            self.dfs.append(pd.read_excel('priceData.xls', sheet_name=sn))
            print("loaded: " + sn)
            self.moves.append([])
            self.wallet.append(0.0)

    def reset(self):
        """Return initial_time_step."""
        self.initial_time_step = ts.restart(self._state)
        self.wallet = [self.initial_balance]

        ef = pd.ExcelFile("priceData.xls")
        for sn in ef.sheet_names:
            self.wallet.append(0.0)
        self.current_step = self.look_back_window

        return ts.restart(self._state)

    def step(self, action):
        """Apply action and return new time_step."""
        data = []
        for df in self.dfs:
            data.append(
                np.array([df['volume'].values[range(self.current_step - self.look_back_window, self.current_step)],
                          df['open'].values[range(self.current_step - self.look_back_window, self.current_step)],
                          df['high'].values[range(self.current_step - self.look_back_window, self.current_step)],
                          df['low'].values[range(self.current_step - self.look_back_window, self.current_step)],
                          df['close'].values[range(self.current_step - self.look_back_window, self.current_step)]]))

        self._state = np.array(data)

        # seperate our action list to different variables
        coin = action[0]
        action_type = action[1]
        amount = action[2] / 10.0

        # initiliaze the reward to 0 for each timestep
        reward = 0

        if self.wallet[0] < 0.01 * self.initial_balance:
            self._episode_ended = True
            return ts.termination(self._state, reward)

        if action_type == 0:
            # Buy coin
            current_price = data[coin][1, self.look_back_window - 1]
            usd_val = amount * self.wallet[0]
            self.wallet[0] -= usd_val
            self.wallet[coin + 1] += usd_val / current_price
            self.moves[coin].append(['buy', self.current_step, current_price])

        if action_type == 1:
            # Sell coin
            current_price = data[coin][1, self.look_back_window - 1]
            coin_val = amount * self.wallet[coin + 1]
            self.wallet[coin + 1] -= coin_val
            self.wallet[0] += coin_val * current_price
            self.moves[coin].append(['sell', self.current_step, current_price])

        # increment the step in our environment
        self.current_step += 1
        # return the timestep as a transition containing the state, reward, and discount value
        return ts.transition(self._state, reward, 1.0)

    def render(self):
        # checks if the visualization object was created
        if self.visual == None:
            self.visual = TradingGraph(self.dfs)
        else:
            # the render function gets called by the agent after each step it takes to replot the graph
            self.visual.render(self.current_step, self.look_back_window, self.moves)

    def current_time_step(self):
        return self._current_time_step

    def observation_spec(self):
        """Return observation_spec."""
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        return self.initial_time_step

    def _step(self, action):
        return self.step(action)


class TradingGraph:
    def __init__(self, dfs, title=None):
        self.dfs = dfs

        # Needed to be able to iteratively update the figure
        plt.ion()

        # Define our subplots
        self.fig, self.axs = plt.subplots(2, 2)

        # Show the plots
        plt.show()

    def plot_candles(self, ax, ohlc, idx):
        # iterate over each ohlc value along with the index
        for row, ix in zip(ohlc, idx):
            if row[3] < row[0]:
                clr = "red"
            else:
                clr = "green"

            # plots a thin line to represent high and low values
            ax.plot([ix, ix], [row[1], row[2]], lw=1, color=clr)
            # plot a wider line to represent open and close
            ax.plot([ix, ix], [row[3], row[0]], lw=3, color=clr)

    def render_prices(self, current_step, lbw):
        for splot, df in zip(self.axs.flatten(), self.dfs):
            splot.clear()

            step_range = range(current_step - lbw, current_step)
            idx = np.array(step_range)

            # prepare a 3-d numpy array to be used by our plot_candles function
            candlesticks = zip(df['open'].values[step_range],
                               df['high'].values[step_range],
                               df['low'].values[step_range],
                               df['close'].values[step_range])

            # Plot price using candlestick graph
            self.plot_candles(splot, candlesticks, idx)

    def render_trades(self, current_step, lbw, trades):
        for splot, coin in zip(self.axs.flatten(), trades):
            for trade in coin:
                if current_step > trade[1] > current_step - lbw:
                    # plot the point at which the trade happened
                    if trade[0] == 'buy':
                        clr = 'red'
                        splot.plot(trade[1], trade[2], 'ro')
                    else:
                        clr = 'green'
                        splot.plot(trade[1], trade[2], 'go')

                else:
                    clr = 'black'

                # the plotted dot won't appear after the look back window is passed so a horizontal line keeps tracks at any time
                splot.hlines(trade[2], current_step - lbw, current_step, linestyle='dashed', colors=[clr])

    def render(self, current_step, window_size, trades):
        self.render_prices(current_step, window_size)
        self.render_trades(current_step, window_size, trades)

        # the draw function redraws the figure after all plots have been executed
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # the pause is necessary to view the frame and allow the live updating mechanism on our figure
        plt.pause(0.1)

    def close(self):
        plt.close()



# initialize our environment
env = CryptoTradingEnv()

for _ in range(0, 100):
    random_action = np.random.choice(np.arange(0, 3), p=[0.1, 0.1, 0.8])
    random_coin = random.randint(0, 3)
    amount = 2
    action_w = np.array([int(random_coin), int(random_action), amount])
    env.step(action_w)
    env.render()