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

#url = 'https://financialmodelingprep.com/api/v3/company/stock/list'

url = 'https://financialmodelingprep.com/api/v3/stock/list?apikey=c247ca711e240e8c07bce1aa1549214d'

ticks_json = get_json_data(url)

#print(ticks_json)

available_tickers2 = find_in_json(ticks_json, 'symbol')

#print(available_tickers2[1])

available_tickers = []
i=0
while i<3000:
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

S = pd.DataFrame(tickers_sector, index=available_tickers, columns=['Sector'])

print('sectors')
print(S)

# Get list of tickers from TECHNOLOGY sector
#tickers_tech = S[S['Sector'] == 'Technology'].index.values.tolist()
tickers_tech = S['Sector'].index.values.tolist()

pvar_list, tickers_found = [], []
#num_tickers_desired = 1000
num_tickers_desired = 8000
count = 0
tot = 0
TICKERS = tickers_tech

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

# Initialize lists and dataframe (dataframe is a 2D numpy array filled with 0s)
missing_tickers, missing_index = [], []
d = np.zeros((len(tickers_found), len(indicators)))

for t, _ in enumerate(tqdm(tickers_found)):
    try:
        # Scrape indicators from financialmodelingprep API
        url0 = 'https://financialmodelingprep.com/api/v3/financials/income-statement/' + tickers_found[t] + accessCode
        url1 = 'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/' + tickers_found[t] + accessCode
        url2 = 'https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/' + tickers_found[t] + accessCode
        url3 = 'https://financialmodelingprep.com/api/v3/financial-ratios/' + tickers_found[t] + accessCode
        url4 = 'https://financialmodelingprep.com/api/v3/company-key-metrics/' + tickers_found[t] + accessCode
        url5 = 'https://financialmodelingprep.com/api/v3/financial-statement-growth/' + tickers_found[t] + accessCode
        a0 = get_json_data(url0)
        a1 = get_json_data(url1)
        a2 = get_json_data(url2)
        a3 = get_json_data(url3)
        a4 = get_json_data(url4)
        a5 = get_json_data(url5)

        # Combine all json files in a list, so that it can be scanned quickly
        A = [a0, a1, a2, a3, a4, a5]
        all_dates = find_in_json(A, 'date')

        check = [s for s in all_dates if '2018' in s]  # find all 2018 entries in dates
        if len(check) > 0:
            date_index = all_dates.index(check[0])  # get most recent 2018 entries, if more are present

            for i, _ in enumerate(indicators):
                ind_list = find_in_json(A, indicators[i])
                try:
                    d[t][i] = ind_list[date_index]
                except:
                    d[t][i] = np.nan  # in case there is no value inserted for the given indicator

        else:
            missing_tickers.append(tickers_found[t])
            missing_index.append(t)
    except:
        print("An exception occurred")

actual_tickers = [x for x in tickers_found if x not in missing_tickers]
d = np.delete(d, missing_index, 0)  # raw dataset
DATA = pd.DataFrame(d, index=actual_tickers, columns=indicators)

# Remove columns that have more than 20 0-values
DATA = DATA.loc[:, DATA.isin([0]).sum() <= 600]

# Remove columns that have more than 15 nan-values
DATA = DATA.loc[:, DATA.isna().sum() <= 600]

# Fill remaining nan-values with column mean value
DATA = DATA.apply(lambda x: x.fillna(x.mean()))

# Get price variation data only for tickers to be used
D2 = D.loc[DATA.index.values, :]

# Generate classification array
y = []
for i, _ in enumerate(D2.index.values):
    #print(D2.values[i])
    if D2.values[i] >= 10:
        y.append(1)
    else:
        y.append(0)

y2 = []
for i, _ in enumerate(D2.index.values):
    #print(D2.values[i])
    if D2.values[i] >= 20:
        y2.append(1)
    else:
        y2.append(0)

y3 = []
for i, _ in enumerate(D2.index.values):
    #print(D2.values[i])
    if D2.values[i] >= 40:
        y3.append(1)
    else:
        y3.append(0)

y4 = []
for i, _ in enumerate(D2.index.values):
    #print(D2.values[i])
    if D2.values[i] >= 80:
        y4.append(1)
    else:
        y4.append(0)

# Add array to dataframe
DATA['class10'] = y
DATA['class20'] = y2
DATA['class40'] = y3
DATA['class80'] = y4

DATA.to_csv('SelfMadeStockDataset.csv')