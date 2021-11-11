from enum import Enum  # Standard Python Library
import time, os, sys  # Standard Python Library
#import xlwings as xw  # pip install xlwings
import pandas as pd  # pip install pandas
from yahoofinancials import YahooFinancials  # pip install yahoofinancials

import yfinance as yf


#data = YahooFinancials('AAPL')
#data = yf.download('AAPL', '2019-01-01', '2019-01-01')

#data = pd.DataFrame(data)

#print(data)
#print(data['Close'])

#data1 = yf.download('AAPL', '2020-01-01', '2020-01-01')
#data1 = pd.DataFrame(data1)

#print(data1)
#print(data1['Close'])

def pull_stock_dataProfitt(ticker, numberOfStocks, datefrom, dateto):
    data1 = YahooFinancials(ticker)
    data1 = yf.download(ticker, datefrom, datefrom)
    data1 = pd.DataFrame(data1)
    data2 = YahooFinancials(ticker)
    data2 = yf.download(ticker, dateto, dateto)
    data2 = pd.DataFrame(data2)

    profit = (data2['Close'].iloc[0]-data1['Close'].iloc[0])*numberOfStocks
    return profit

profitten =pull_stock_dataProfitt('AAPL', 3, '2019-01-01', '2020-01-01')

print(profitten)
