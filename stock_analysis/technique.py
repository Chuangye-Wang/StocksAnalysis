import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

def ticker_SMA(Ticker, periods = 200, moving_object = 'Adj Close', Add_SMA = True):
    """
    Compute the moving average of stock price for one ticker with specified periods.

    Ticker: an object of class Ticker.
    periods: an integer indicating the periods for computing moving average, default is 200.
    moving_object: a kind of stock price to calculate, like 'High', 'Low'...
    the default is 'Adj Close'

    Returns: SMA, a Series
    """
    # TODO
    
    new_feature = 'SMA ' + str(periods)
    y = Ticker.get_ticker_data()[moving_object]
    y_SMA = []
    y_len = len(y)
    if periods > y_len:
        raise ValueError('The {} periods is longer than the stocks total trading days {}, please enlarge the date range or maker periods smaller'.format(periods, y_len))
    if new_feature not in Ticker.get_ticker_data().columns:
        for i in range(0, periods-1):
            y_SMA.append(np.nan)
        for j in range(periods-1, y_len):
            y_SMA.append(np.mean(y[j+1-periods: j+1]))
        if Add_SMA:   
            Ticker.get_ticker_data()[new_feature] = y_SMA #'moving average' is a new feature for this Ticker.
            
        return pd.Series(y_SMA, index = Ticker.get_ticker_data().index)
    
    return Ticker.get_ticker_data()[new_feature]
  

def ticker_SMA_plot(Ticker, periods = 200, plot_content = 'Adj Close', Add_SMA = True):
    """
    Compute the moving average of stock price for one ticker with specified periods.

    Tickers: a dictionary contains objects of class Ticker.
    periods: an integer indicating the periods for computing moving average, default is 200.
    plot_content: a string, the stock price data that you want to show in a figure.
    The plot_content is also the moving_object.

    Returns: no
    """
    mv_object = plot_content
#    new_feature = 'SMA ' + str(periods)
    t_SMA = ticker_SMA(Ticker, periods, mv_object, Add_SMA = Add_SMA)
    t_SMA.plot(color = 'black', label = '{}, SMA, {}'.format(Ticker.get_ticker_name(), periods))
    
    plt.xlim([Ticker.get_ticker_data().index[0], Ticker.get_ticker_data().index[-1]])
    leg = plt.legend(fontsize = 8,  loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)
    
def tickers_SMA_plot(Tickers, periods = 200, plot_content = 'Adj Close', Add_SMA = True):
    """
    Compute the moving average of stock price for tickers with specified periods.

    Tickers: a dictionary contains objects of class Ticker.
    periods: an integer indicating the periods for computing moving average, default is 200.
    plot_content: a string, the stock price data that you want to show in a figure.
    The plot_content is also the moving_object.

    no returns
    """
    mv_object = plot_content
#    new_feature = 'SMA ' + str(periods)
    tickers_Series = [0]*len(Tickers)
    for Ticker_name in Tickers:
        i = 0
        tickers_Series[i] = ticker_SMA(Tickers[Ticker_name], periods, mv_object, Add_SMA = Add_SMA)
        i += 1
    for i in range(0, len(Tickers)):
        tickers_Series[i].plot(label = '{}, SMA, {}'.format(Ticker_name, periods))
    
    plt.xlim([Ticker.get_ticker_data().index[0], Ticker.get_ticker_data().index[-1]])
    leg = plt.legend(fontsize = 8,  loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)

def ticker_EMA(Ticker, periods = 20, smoothing = 2, moving_object = 'Adj Close', Add_EMA = True):
    """
    Compute the exponential moving average of stock price for one ticker with specified periods.

    Ticker: an object of class Ticker.
    periods: an integer indicating the periods for computing moving average, default is 200.
    moving_object: a kind of stock price to calculate, like 'High', 'Low'...
    the default is 'Adj Close'

    Returns: exponential moving average, a Series.
    """
    
    y = Ticker.get_ticker_data()[moving_object]
    y_EMA = []
    y_len = len(y)
    new_feature = 'EMA ' + str(periods)
    
    if periods > y_len:
        raise ValueError('The {} periods is longer than the stocks total trading days {}, please enlarge the date range or maker periods smaller'.format(periods, y_len))
    if new_feature not in Ticker.get_ticker_data().columns:
        for i in range(0, periods - 1):
            y_EMA.append(np.nan)
            
        y_EMA.append(np.mean(y[0:periods])) # add y_EMA[periods - 1] to y_EMA
    
        for j in range(periods, y_len):
            item_1 = y[j] * smoothing / (1+periods)
            item_2 = y_EMA[j-1] * (1 - smoothing/(1+periods))
            y_EMA.append(item_1 + item_2)
        if Add_EMA:
            Ticker.get_ticker_data()[new_feature] = y_EMA
        
        return pd.Series(y_EMA, index = Ticker.get_ticker_data().index)
        
    return Ticker.get_ticker_data()[new_feature]


def ticker_EMA_plot(Ticker, periods = 200, plot_content = 'Adj Close', Add_EMA = True):
    """
    Compute the exponential moving average of stock price for one ticker with specified periods.

    Tickers: a dictionary contains objects of class Ticker.
    periods: an integer indicating the periods for computing moving average, default is 200.
    plot_content: a string, the stock price data that you want to show in a figure.
    The plot_content is also the moving_object.

    Returns: no
    """
    mv_object = plot_content
#    new_feature = 'EMA ' + str(periods)
    t_EMA = ticker_EMA(Ticker, periods, moving_object = mv_object, Add_EMA = Add_EMA)
    t_EMA.plot(label = '({}, EMA, {})'.format(Ticker.get_ticker_name(), periods))
    
    plt.xlim([Ticker.get_ticker_data().index[0], Ticker.get_ticker_data().index[-1]])
    leg = plt.legend(fontsize = 8,  loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)
        

def tickers_EMA_plot(Tickers, periods = 200, plot_content = 'Adj Close', Add_EMA = True):
    """
    Compute the moving average of stock price for tickers with specified periods.

    Tickers: a dictionary contains objects of class Ticker.
    periods: an integer indicating the periods for computing moving average, default is 200.
    plot_content: a string, the stock price data that you want to show in a figure.
    The plot_content is also the moving_object.

    no returns
    """
    mv_object = plot_content
#    new_feature = 'SMA ' + str(periods)
    tickers_Series = [0]*len(Tickers)
    for Ticker_name in Tickers:
        i = 0
        tickers_Series[i] = ticker_EMA(Tickers[Ticker_name], periods, mv_object, Add_EMA = Add_EMA)
        i += 1
    for i in range(0, len(Tickers)):
        tickers_Series[i].plot(label = '{}, SMA, {}'.format(Ticker_name, periods))
    
    plt.xlim([Ticker.get_ticker_data().index[0], Ticker.get_ticker_data().index[-1]])
    leg = plt.legend(fontsize = 8,  loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)
        

def ticker_RSI_plot(Ticker, periods = 14, threshold = 30, RSI_type = 1, Add_RSI = True):
    """
    Compute the Relative Strength Index (RSI) of stock price for one ticker with specified periods.

    Tickers: a dictionary contains objects of class Ticker.
    periods: an integer indicating the periods for computing moving average, default is 14.
    Threshold: the relisience level, default is 30, which means it is oversold when RSI < 30.
    RSI_type: 1 -> non_smooth, 2 -> smooth
    
    no returns
    """
    
    if threshold < 0 or threshold > 100:
        raise ValueError('threshould must be in range [0, 100]')
    if threshold <= 50:
        threshold_low = threshold
        threshold_high = 100 - threshold
    else:
        threshold_low = 100 - threshold
        threshold_high = threshold
        
    new_feature = 'RSI ' + str(periods)   
    y = np.array(Ticker.get_ticker_data()['Adj Close'])
    y_len = len(y)
    if periods > y_len:
        raise ValueError('The {} periods is longer than the stocks total trading days {}, please enlarge the date range or maker periods smaller'.format(periods, y_len))
    if new_feature not in Ticker.get_ticker_data().columns:
        RSI = []
        for i in range(0, periods):
            RSI.append(np.nan)
            
        if RSI_type == 1:
            for j in range(1, y_len - periods + 1):
                y_right = y[j : j+periods]
                y_left = y[j-1 : j+periods-1]
                y_ratio = y_right/y_left - 1
                y_ratio_pos = y_ratio[y_ratio > 0]
                y_ratio_neg = -y_ratio[y_ratio < 0]
                if len(y_ratio_pos) == 0:
                    RS = 0
                elif len(y_ratio_neg) == 0:
                    RS = 100
                else:
                    average_gain = sum(y_ratio_pos) # / periods
                    average_loss = sum(y_ratio_neg) # / periods
                    RS = 100 - 100/(1 + average_gain/average_loss)
                RSI.append(RS)
                
        if RSI_type == 2:
            for j in range(1, y_len - periods + 1):
                y_right = y[j : j+periods]
                y_left = y[j-1 : j+periods-1]
                y_ratio = y_right/y_left - 1
                y_ratio_pos = y_ratio[y_ratio > 0]
                y_ratio_neg = -y_ratio[y_ratio < 0]
                if len(y_ratio_pos) == 0:
                    RS = 0
                elif len(y_ratio_neg) == 0:
                    RS = 100
                else:
                    previous_average_gain = sum(y_ratio_pos) / periods
                    previous_average_loss = sum(y_ratio_neg) / periods
                    RS = 100 - 100/(1 + (previous_average_gain*13 + y_ratio_pos[-1])/(previous_average_loss*13 + y_ratio_neg[-1]))
                RSI.append(RS)
        if Add_RSI:    
            Ticker.get_ticker_data()[new_feature] = RSI
    if Add_RSI:
        Ticker.get_ticker_data()[new_feature][periods:].plot(color = 'violet', linewidth = 0.6, label = '{}, {}'.format(Ticker.get_ticker_name(), new_feature))
    else:
        pd.Series(RSI, index = Ticker.get_ticker_data().index).plot(color = 'violet', linewidth = 0.6, label = '{}, {}'.format(Ticker.get_ticker_name(), new_feature))
    leg = plt.legend(fontsize = 8, loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)
    plt.ylim([0, 100])
    plt.yticks([0, threshold_low, 50, threshold_high, 100])
    plt.xlim([Ticker.get_ticker_data().index[0], Ticker.get_ticker_data().index[-1]])
    plt.axhline(y=threshold_low, xmin=0, xmax=1, color = 'cyan', linestyle = '-', linewidth = 0.5)
    plt.axhline(y=threshold_high, xmin=0, xmax=1, color = 'cyan', linestyle = '-', linewidth = 0.5)
    
    RSI_no_nan = np.array(RSI[periods:])
    y_low = np.ones(y_len - periods)*threshold_low
    y_high = np.ones(y_len - periods)*threshold_high
    date_x = Ticker.get_ticker_data().index[periods:]
    plt.fill_between(date_x, RSI_no_nan, y_high, where=RSI_no_nan > y_high, facecolor='pink', interpolate = True)
    plt.fill_between(date_x, RSI_no_nan, y_low, where=RSI_no_nan < y_low, facecolor='pink', interpolate = True)
    ypos_high = threshold_high/100 + 0.05
    ypos_low = threshold_low/100 - 0.1
    plt.annotate('overbought > {}%'.format(threshold_high), xy=(0.75, ypos_high), fontsize = 8, xycoords='axes fraction')
    plt.annotate('oversold < {}%'.format(threshold_low), xy=(0.75, ypos_low), fontsize = 8, xycoords='axes fraction')
    

def ticker_Bollinger_Bands_plot(Ticker, m = 2, periods = 21, Add_BB = True):
    """
    To compute the Bollinger Bands of a stock
    
    Tickers: a dictionary contains objects of class Ticker.
    periods: an integer indicating the periods for computing moving average, typically is 20.
    m: an interger used to control the band size which fluctuates within m*standard deviation
    
    no returns
    """
    new_BOLU = 'BOLU ' + str(periods)
    new_BOLM = 'BOLM ' + str(periods)
    new_BOLD = 'BOLD ' + str(periods)
    ticker_data = Ticker.get_ticker_data()
    tp = (ticker_data['Adj Close'] + ticker_data['Low'] + ticker_data['High']) / 3  # tp: typical price
    BOLU = []
    BOLM = []
    BOLD = []
    y_len = len(tp)
    if periods > y_len:
        raise ValueError('The {} periods is longer than the stocks total trading days {}, please enlarge the date range or maker periods smaller'.format(periods, y_len))
    if new_BOLU not in Ticker.get_ticker_data().columns:    
        for i in range(0, periods-1):
            BOLU.append(np.nan)
            BOLM.append(np.nan)
            BOLD.append(np.nan)
        for j in range(periods-1, y_len):
            tp_mean = np.mean(tp[j+1-periods: j+1])
            tp_std = np.std(tp[j+1-periods: j+1])
            BOLU.append(tp_mean + m*tp_std)
            BOLM.append(tp_mean)
            BOLD.append(tp_mean - m*tp_std)
        if Add_BB:
            ticker_data[new_BOLU] = BOLU
            ticker_data[new_BOLM] = BOLM    
            ticker_data[new_BOLD] = BOLD
    
    date_x = ticker_data.index
    if Add_BB:
        ticker_data[new_BOLU].plot(color = 'green', linewidth = 0.6, label = '')
        ticker_data[new_BOLM].plot(color = 'yellow', linewidth = 0.6, label = '')
        ticker_data[new_BOLD].plot(color = 'red', linewidth = 0.6, label = '')
    else :
        pd.Series(BOLU, index = date_x).plot(color = 'green', linewidth = 0.6, label = '')
        pd.Series(BOLM, index = date_x).plot(color = 'yellow', linewidth = 0.6, label = '')
        pd.Series(BOLD, index = date_x).plot(color = 'red', linewidth = 0.6, label = '')
    
    plt.fill_between(date_x, BOLU, BOLD, facecolor='mistyrose', label = 'Bollinger Bands', interpolate = True)
    
    leg = plt.legend(fontsize = 8, loc = 'upper left') # , facecolor='white', framealpha=0
    if leg:
        leg.set_draggable(state=True)
#    plt.annotate('Bollinger Bands', xy=(0.1, 0.8), fontsize = 8, xycoords='axes fraction')


def ticker_MACD_plot(Ticker, short_periods = 12, long_periods = 26, signal_periods = 9, Add_MACD = True):
    """
    To compute the Moving Average Convergence Divergence (MACD) of a stock
    
    Tickers: a dictionary contains objects of class Ticker.
    short_periods: an integer, a period for calculating short-period moving average, typically is 12.
    long_periods: an integer, a period for calculating long-period moving average, typically is 26.
    signal_periods: an integer, a period for calculating signal-period moving average for MACD, typically is 26.
    Add_MACD: boolean value, True to add MACD data to this Ticker.
    
    no returns
    """
    pass
    if short_periods >= long_periods:
        raise ValueError("The short period is greater than long period now, please reinput these two number")
    new_shortp = 'MACD ' + str(short_periods)
    new_longp = 'MACD ' + str(long_periods)
    new_signalp = 'MACD ' + str(signal_periods)
    new_diffp = 'MACD ' + '{}-{}-{}'.format(short_periods, long_periods, signal_periods)
    y = Ticker.get_ticker_data()['Adj Close']
    y_len = len(y)
    if short_periods > y_len:
        raise ValueError('The shot periods {} is longer than the stocks total trading days {}, please enlarge the date range or maker periods smaller'.format(short_periods, y_len))
    columns = Ticker.get_ticker_data().columns
    if new_shortp not in columns:    
        shortp = ticker_EMA(Ticker, periods = short_periods, Add_EMA = False)
    if new_longp not in columns:    
        longp = ticker_EMA(Ticker, periods = long_periods, Add_EMA = False)
    if new_diffp not in columns:
        sl_MACD = shortp - longp
        if Add_MACD:
            Ticker.get_ticker_data()[new_diffp] = sl_MACD   
    signalp = sl_MACD.copy() 
    if new_signalp not in columns:
        start = long_periods + signal_periods
        signalp[start - 1] = np.mean(signalp[long_periods: start])
        for j in range(start, y_len):
            item_1 = signalp[j] * 2 / (1+signal_periods)
            item_2 = signalp[j-1] * (1 - 2/(1+signal_periods))
            signalp[j] = item_1 + item_2
        signalp[0:long_periods + signal_periods] = np.nan
        if Add_MACD:
            Ticker.get_ticker_data()[new_signalp] = signalp   
    else:
        signalp = Ticker.get_ticker_data()[new_signalp]

    sl_MACD.plot(color = 'green', linewidth = 1, label = 'MACD, {}, {}, {}'.format(short_periods, long_periods, signal_periods))
    signalp.plot(color = 'orange', linewidth = 1, label = 'signal line')
    
    leg = plt.legend(fontsize = 8, loc = 'upper left') # , facecolor='white', framealpha=0
    if leg:
        leg.set_draggable(state=True)
        

def ticker_return_rate(Ticker):
    """
    Ticker: an object of class Ticker.
    
    return: return rate in pd.Series
    """
    if 'Return Rate' not in Ticker.get_ticker_data().columns:
        return_rate = Ticker.get_ticker_data()['Adj Close']/Ticker.get_ticker_data()['Adj Close'][0]
        Ticker.get_ticker_data()['Return Rate'] = return_rate
    
    return return_rate


def ticker_return_rate_plot(Ticker):
    """
    Ticker: an object of class Ticker.
    
    no returns
    """
    new_feature = 'Return Rate'
    ticker_return_rate(Ticker)
    Ticker.get_ticker_data()[new_feature].plot(label = 'RR of {}'.format(Ticker.get_ticker_name()))
    leg = plt.legend(fontsize = 8, loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)
    plt.ylabel('Return Rate in one stock')

        
def tickers_return_rate_plot(Tickers):
    """
    Tickers: a dictionary contains objects of class Ticker.
    """
    new_feature = 'Return Rate'
    
    for Ticker_name in Tickers:
        ticker_return_rate(Tickers[Ticker_name])
    for Ticker_name in Tickers:
        Tickers[Ticker_name].get_ticker_data()[new_feature].plot(label = 'RR of {}'.format(Ticker_name))
    leg = plt.legend(fontsize = 8, loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)
    plt.ylabel('Return Rate in one stock')

    
def return_rate_of_portfolio(Tickers, weights):
    """
    Tickers: a dictionary contains objects of class Ticker.
    weights: a dictionary contains your portfolio for every stock
    
    return: a pd.series, return rate of a portfolio
    """
    RR_of_portfolio = 0
    for Ticker_name in Tickers:
        ticker_return_rate(Tickers[Ticker_name])
        RR_of_portfolio += weights[Ticker_name] * Tickers[Ticker_name].get_ticker_data()['Return Rate']
    
    return RR_of_portfolio
       

def portfolio_return_plot(Tickers, weights):
    """
    To visualize a portfolio as time goes.
    
    Tickers: a dictionary contains objects of class Ticker.
    weights: a dictionary contains your portfolio for every stock
    
    no return
    """
    if sum(weights.values()) != 1:
        raise ValueError('The total weight is not 1 here, please change your portfolio weights')
    RR_portf = return_rate_of_portfolio(Tickers, weights)
    RR_portf.plot(label = 'RR of portfolio')
    leg = plt.legend(fontsize = 8, loc = 'upper left')
    if leg:
        leg.set_draggable(state=True)
    plt.ylabel('Return Rate in one stock')


#def tickers_volume_plot(Tickers):
#    """
#    Tickers: a dictionary contains Tickers (class)
#    
#    no return
#    """
#    # the default figsize = (6.4, 4.8)
#    for Ticker_name in Tickers:
#        Tickers[Ticker_name].get_ticker_data()['Volume'].plot.bar(label = Ticker_name)
#    leg = plt.legend()
#    if leg:
#        leg.set_draggable(state=True)
#    plt.ylabel('Volume')