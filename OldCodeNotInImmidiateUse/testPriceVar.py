from sys import stdout
import numpy as np
import pandas as pd
from pandas_datareader import data
import json

# Reading data from external sources
import urllib as u
from urllib.request import urlopen

# Machine learning (preprocessing, models, evaluation)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import classification_report

def get_price_var(symbol):
    '''
    Get historical price data for a given symbol leveraging the power of pandas_datareader and Yahoo.
    Compute the difference between first and last available time-steps in terms of Adjusted Close price..
    Input: ticker symbol
    Output: price variation
    '''
    # read data
    prices = data.DataReader(symbol, 'yahoo', '2019-01-01', '2019-12-31')['Adj Close']

    # get all timestamps for specific lookups
    today = prices.index[-1]
    start = prices.index[0]

    # calculate percentage price variation
    price_var = ((prices[today] - prices[start]) / prices[start]) * 100
    return price_var

resultat = get_price_var('NYMX')
print(resultat)

resultat = get_price_var('IBM')
print(resultat)

dataset = pd.read_csv("SelfMadeStockDataset.csv")

X = dataset.iloc[:, 0:1]

list = X.values.tolist()

#print(list)

list2 = []

#X = X.index.values.tolist()

tot = 0
for x in list:
    tot += 1
    list2.append(x[0])

    stdout.write(f'\rScanned {tot} tickers.')
    stdout.flush()

#print(list2)

print(list2)


#dataPriceVar = list(data[])

#print(dataPriceVar)

#for row in data.index:
  #  print(row, end = " ")

dataPriceVarAfterGet = []


pvar_list, tickers_found = [], []
#num_tickers_desired = 1000
num_tickers_desired = 8000
count = 0
tot = 0
TICKERS = list2

#D = pd.DataFrame(pvar_list, index=dataPriceVar, columns=['2019 PRICE VAR [%]'])

for ticker in TICKERS:
    tot += 1
    try:
        pvar = get_price_var(ticker)
        pvar_list.append(pvar)
        tickers_found.append(ticker)
        count += 1
    except:
        pass

    stdout.write(f'\rScanned {tot} tickers. Found {count}/{len(TICKERS)} usable tickers (max tickets = {num_tickers_desired}).')
    stdout.flush()

    if count == num_tickers_desired: # if there are more than 1000 tickers in sectors, stop
        break

# Store everything in a dataframe
D = pd.DataFrame(pvar_list, index=tickers_found, columns=['2019 PRICE VAR [%]'])

D.to_csv('StockTickersAndPriceVar.csv')

