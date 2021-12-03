from sys import stdout
import numpy as np
import pandas as pd
from pandas_datareader import data
import json

# Reading data from external sources
import urllib as u
from urllib.request import urlopen


def get_price_var(symbol):
    '''
    Get historical price data for a given symbol leveraging the power of pandas_datareader and Yahoo.
    Compute the difference between first and last available time-steps in terms of Adjusted Close price..
    Input: ticker symbol
    Output: price variation
    '''
    # read data
    prices = data.DataReader(symbol, 'yahoo', '2020-01-01', '2020-12-31')['Adj Close']

    # get all timestamps for specific lookups
    today = prices.index[-1]
    start = prices.index[0]

    # calculate percentage price variation
    price_var = ((prices[today] - prices[start]) / prices[start]) * 100
    return price_var


AllTickersCleanListlist = pd.read_csv('ALLTickersClean.csv')
print('printer dataframen')
print(AllTickersCleanListlist)
#pvar_list = pvar_list.to_numpy().tolist()
AllTickersCleanListlist = AllTickersCleanListlist['0'].tolist()


# Get list of tickers from TECHNOLOGY sector
#tickers_tech = S[S['Sector'] == 'Technology'].index.values.tolist()
tickers_tech = AllTickersCleanListlist


pvar_list, tickers_found, tickerSector = [], [], []
#num_tickers_desired = 1000
num_tickers_desired = 50000
count = 0
tot = 0
TICKERS = tickers_tech

for ticker in TICKERS:
    tot += 1
    try:
        pvar = get_price_var(ticker)
        pvar_list.append(pvar)
        tickers_found.append(ticker)
        #tickerSector.append(S.loc[ticker])
        count += 1
    except:
        pass

    stdout.write(f'\rScanned {tot} tickers. Found {count}/{len(TICKERS)} usable tickers (max tickets = {num_tickers_desired}).')
    stdout.flush()

    if count == num_tickers_desired: # if there are more than 1000 tickers in sectors, stop
        break



# Store everything in a dataframe
#D = pd.DataFrame(pvar_list, index=tickers_found, columns=['2019 PRICE VAR [%]'])

pvardf = pd.DataFrame(pvar_list)
pvardf.to_csv('pvarFromTicketBuilder.csv')

tickers_founddf = pd.DataFrame(tickers_found)
tickers_founddf.to_csv('tickers_foundFromTicketBuilder.csv')


#print('printer ut dfen')
#print(tickerSectordf['Sector'])