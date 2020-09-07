import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
# import time
#import warnings

class Ticker():
    def __init__(self, name):
        """
        name: string type, the name of a ticker.
        """
        self.name = name
        self.ticker_data = None
        
    def get_ticker_name(self):
        return self.name
        
    def get_ticker_data_from_web(self, start_date = datetime.datetime(1970, 1, 1), end_date = 'today', source = 'yahoo'):
        """
        source: string type, the source of ticker's data. 'yahoo' is the default.
        start_date: a string, starting date in the history of this ticker, the date should be in y-m-d form.
        end_date: a string, ending date in the history of this ticker, the date should be in y-m-d form.
        """
        if end_date == 'today':
            self.ticker_data = wb.DataReader(self.get_ticker_name(), data_source=source, start = start_date)
        else:
            self.ticker_data = wb.DataReader(self.get_ticker_name(), data_source=source, start = start_date, end = end_date)
    
    def get_ticker_data(self):
        """
        return the history data of this ticker.
        """
        if self.ticker_data is None:
            raise ValueError('Please get ticker data from website first')
            
        return self.ticker_data


def make_tickers_to_class(tickers):
    """
    tickers: a list contains the names of tickers (string)
    
    returns: a dictionary contains different classes of Tickers
    """
    dict0_Tickers = {}
    for ticker in tickers:
        dict0_Tickers[ticker] = Ticker(ticker)
    
    return dict0_Tickers

def get_data_for_all_tickers(Tickers, start_date, end_date):
    """
    Tickers: a dictionary contains different Tickers (class)
    start_date: a string, starting date in the history of this ticker, the date should be in y-m-d form.
    end_date: a string, ending date in the history of this ticker, the date should be in y-m-d form.
    
    Returns: no return
    """
    for Ticker_name in Tickers:
        Tickers[Ticker_name].get_ticker_data_from_web(start_date = start_date, end_date = end_date)
