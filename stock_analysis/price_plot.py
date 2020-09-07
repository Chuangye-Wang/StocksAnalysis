import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

def ticker_price_plot(Ticker, plot_content = 'Adj Close'):
    """
    Ticker: an Ticker object(class)
    start_date: a string, starting date in the history of this ticker, the date should be in y-m-d form.
    end_date: a string, ending date in the history of this ticker, the date should be in y-m-d form.
    plot_content: a string, the stock price data that you want to show in a figure. The default is 'Adj Close'
    
    no return
    """
    # the default figsize = (6.4, 4.8)
    Ticker.get_ticker_data()[plot_content].plot(label = Ticker.get_ticker_name())
    leg = plt.legend(fontsize = 8, loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)
    plt.ylabel('Stock Price')
    
    
def tickers_price_plot(Tickers, plot_content = 'Adj Close'):
    """
    Tickers: a dictionary contains Tickers (class)
    start_date: a string, starting date in the history of this ticker, the date should be in y-m-d form.
    end_date: a string, ending date in the history of this ticker, the date should be in y-m-d form.
    plot_content: a string, the stock price data that you want to show in a figure. The default is 'Adj Close'
    
    no return
    """
    # the default figsize = (6.4, 4.8)
    for Ticker_name in Tickers:
        Tickers[Ticker_name].get_ticker_data()[plot_content].plot(label = Ticker_name)
    leg = plt.legend(fontsize = 8, loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)
    plt.ylabel('Stock Price')