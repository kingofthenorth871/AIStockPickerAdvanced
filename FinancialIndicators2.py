import pickle

from sys import stdout
import numpy as np
import pandas as pd
from pandas_datareader import data
import json

from tqdm import tqdm

# Reading data from external sources
import urllib as u
from urllib.request import urlopen

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

filename = 'pickleList'
infile = open(filename,'rb')
A2 = pickle.load(infile)
infile.close()


missing_tickers, missing_index = [], []

pvar_list = pd.read_csv('pvarFromTicketBuilder.csv')
print('printer dataframen')
print(pvar_list)
#pvar_list = pvar_list.to_numpy().tolist()
pvar_list = pvar_list['0'].tolist()
#pvar_list = pvar_list[0]
print('printer dataframen etter konverteringen')
print(pvar_list)

tickers_found = pd.read_csv('tickers_foundFromTicketBuilder.csv')
print('printer dataframen')
print(tickers_found)
tickers_found = tickers_found['0'].tolist()
print('printer dataframen etter konverteringen')
print(tickers_found)

a_file = open("indicators.txt")
file_contents = a_file.read()
indicators = file_contents.splitlines()

d = np.zeros((len(tickers_found), len(indicators)))
D = pd.DataFrame(pvar_list, index=tickers_found, columns=['2019 PRICE VAR [%]'])


tickerNumber = 0
for A in A2:
    #try:
        all_dates = find_in_json(A, 'date')

        tqdm(tickerNumber)

        t = tickerNumber

        tickerNumber = tickerNumber + 1

        pd.DataFrame(pvar_list, index=tickers_found, columns=['2019 PRICE VAR [%]'])
        #testAllFinancialData = pd.DataFrame(A)
        #testAllFinancialData.to_csv('testAllFinancialData.csv')

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
    #except:
        #print("An exception occurred")

actual_tickers = [x for x in tickers_found if x not in missing_tickers]
d = np.delete(d, missing_index, 0)  # raw dataset
DATA = pd.DataFrame(d, index=actual_tickers, columns=indicators)

# Remove columns that have more than 20 0-values
# DATA = DATA.loc[:, DATA.isin([0]).sum() <= 600]

# Remove columns that have more than 15 nan-values
# DATA = DATA.loc[:, DATA.isna().sum() <= 600]

# Fill remaining nan-values with column mean value
# DATA = DATA.apply(lambda x: x.fillna(x.mean()))

# Get price variation data only for tickers to be used
D2 = D.loc[DATA.index.values, :]

# Generate classification array
y = []
for i, _ in enumerate(D2.index.values):
    # print(D2.values[i])
    if D2.values[i] >= 10:
        y.append(1)
    else:
        y.append(0)

y2 = []
for i, _ in enumerate(D2.index.values):
    # print(D2.values[i])
    if D2.values[i] >= 20:
        y2.append(1)
    else:
        y2.append(0)

y3 = []
for i, _ in enumerate(D2.index.values):
    # print(D2.values[i])
    if D2.values[i] >= 40:
        y3.append(1)
    else:
        y3.append(0)

y4 = []
for i, _ in enumerate(D2.index.values):
    # print(D2.values[i])
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