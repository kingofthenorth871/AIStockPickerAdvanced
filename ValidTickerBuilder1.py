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

# Graphics
from tqdm import tqdm

def get_json_data(url):
    '''
    Scrape data (which must be json format) from given url
    Input: url to financialmodelingprep API
    Output: json file
    '''
    response = urlopen(url)
    dat = response.read().decode('utf-8')
    return json.loads(dat)

def find_in_json(obj, key):
    '''
    Scan the json file to find the value of the required key.
    Input: json file
           required key
    Output: value corresponding to the required key
    '''
    # Initialize output as empty
    arr = []

    def extract(obj, arr, key):
        '''
        Recursively search for values of key in json file.
        '''
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

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

#url = 'https://financialmodelingprep.com/api/v3/company/stock/list'

url = 'https://financialmodelingprep.com/api/v3/stock/list?apikey=c247ca711e240e8c07bce1aa1549214d'

ticks_json = get_json_data(url)

#print(ticks_json)

available_tickers2 = find_in_json(ticks_json, 'symbol')

#print(available_tickers2[1])

available_tickers = []
i=0
while i<len(available_tickers2):
#while i < 300:
    #print(i)
    available_tickers.append(available_tickers2[i])
    i=i+1

# Import Module
import os

# Folder Path
#path = ""

# Change the directory
#os.chdir(path)


# Read text File


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        (f.read())


a_file = open("indicators.txt")
file_contents = a_file.read()
indicators = file_contents.splitlines()

#print('indicators length')
#print(len(indicators))
#print(indicators)


accessCode = '?apikey=c247ca711e240e8c07bce1aa1549214d'

apikeyquestion = ''

#https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=YOUR_API_KEY

tickers_sector = []
for tick in tqdm(available_tickers):
    url = 'https://financialmodelingprep.com/api/v3/profile/' + tick + accessCode # get sector from here
    #print(url)

    try:
        a = get_json_data(url)
        tickers_sector.append(find_in_json(a, 'sector'))
    except:
        print("An exception occurred")

#print('printer ut tickers_sector')
#tickers_sector = tickers_sector
#print(tickers_sector)
tickerSectorClean = []
for sectorEntry in tickers_sector:
    tickerSectorClean.append(sectorEntry[0])


print('printer ut tickers_sectorClean')
print(tickerSectorClean)

S = pd.DataFrame(tickers_sector, index=available_tickers, columns=['Sector'])

print('printer sectors')
print(S)

# Get list of tickers from TECHNOLOGY sector
#tickers_tech = S[S['Sector'] == 'Technology'].index.values.tolist()
tickers_tech = S.index.values.tolist()

tickers_tickers_tech = pd.DataFrame(tickers_tech)
tickers_tickers_tech.to_csv('ALLTickersClean.csv')

tickerSectordf = pd.DataFrame(tickerSectorClean)
tickerSectordf.to_csv('tickerSectorFromTicketBuilder.csv')