import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import time
from stock_analysis.core import Ticker, make_tickers_to_class, 
get_data_for_all_tickers 
from stock_analysis.price_plot import ticker_price_plot, tickers_price_plot
from stock_analysis.technique import ticker_SMA, ticker_SMA_plot, tickers_SMA_plot,
ticker_EMA, ticker_EMA_plot, tickers_EMA_plot, ticker_RSI_plot,
ticker_Bollinger_Bands_plot, ticker_MACD_plot


#%%
if __name__ == "__main__":

    start = datetime.datetime(2019, 1, 1)
    end = datetime.datetime(2020, 1, 4)
#-------- tickers test -----------

#    current_time = datetime.datetime.now()
#    today = datetime.datetime(current_time.year, current_time.month, current_time.day, 0, 0)
#    Algn = Ticker('ALGN')
#    Algn.get_ticker_data_from_web()
#    Algn_data = Algn.get_ticker_data()
#    tickers = ['AAPL', 'ALGN', 'MSFT']

#    tickers = ['ALGN', 'EHTH', 'DAL', 'ANTM']
#    weights = {'ALGN': 0.2, 'EHTH': 0.4, 'DAL': 0.2, 'ANTM': 0.2}
#    dict_Tickers = make_tickers_to_class(tickers)
#    get_data_for_all_tickers(dict_Tickers, start, 'today')
#
#    plt.figure(figsize = (8, 6))
#    tickers_price_plot(dict_Tickers)
#    tickers_moving_average_plot(dict_Tickers, periods = 50)
#    
#    plt.figure(figsize = (8, 6))
#    tickers_return_rate_plot(dict_Tickers)
##    RR = return_rate_of_portfolio(dict_Tickers, weights)
#    portfolio_return_plot(dict_Tickers, weights)


#-------- ticker test -----------

    Ticker_algn = Ticker('ALGN')  #^DJI
    Ticker_algn.get_ticker_data_from_web(start, 'today')
#
#    plt.figure(figsize = (8, 6))
##    tickers_volume_plot(dict_Tickers1)
#    ticker_price_plot(Ticker_algn)
#    ticker_moving_average_plot(Ticker_algn)
#    
#    plt.figure(figsize = (8, 6))
#    ticker_return_rate_plot(Ticker_algn)
#    

    fig, ax= plt.subplots(3,1, figsize = (10, 6))
#    legend = plt.legend(frameon = 1)
#    frame = legend.get_frame()
#    frame.set_color('white')
    mpl.rcParams['legend.frameon'] = 'True'
#    plt.figure(figsize = (8, 6))
    plt.subplot(311)
    ticker_price_plot(Ticker_algn)
    ticker_Bollinger_Bands_plot(Ticker_algn, Add_BB = False)
    ticker_SMA_plot(Ticker_algn, periods = 50, Add_SMA = False)
#    ticker_EMA_plot(Ticker_algn, periods = 50)
#    ticker_SMA_plot(Ticker_algn, periods = 50)
#    plt.tick_params(labelbottom=False)
    plt.tick_params(axis = 'x', which = 'both', bottom = True, labelbottom=False)
    plt.xlabel('')
    # plt.axis('off')
    # plt.axes().axes.get_xaxis().set_visible(False)
    
    start_time = time.time()
    plt.subplot(312)
    ticker_MACD_plot(Ticker_algn, Add_MACD = False)
    plt.tick_params(axis = 'x', which = 'both', bottom = True, labelbottom=False)
    plt.xlabel('')
    end_time = time.time()
    print('running time = ', end_time - start_time)
    
#    sma = ticker_SMA(Ticker_algn, periods = 50)
#    algn_sma = ticker_SMA(Ticker_algn)
    plt.subplot(313)
    ticker_RSI_plot(Ticker_algn, periods = 14, RSI_type = 2, Add_RSI = False)
   