from glob import glob
import pandas as pd

def printAllRelevantNumbers(dataTickersFromTransactionBuilder):

    last_file=glob(dataTickersFromTransactionBuilder)[-1]
    all_transactions = pd.read_excel(last_file)

    all_tickers = list(all_transactions['ticker'].unique())

    total = all_transactions['gain_loss'].sum()
    print(total)

    print(len(all_tickers))

    print(total/len(all_tickers))

    print((total/len(all_tickers))/5000)

    #(AllPrices.loc[date:date]['Close'][0]==0)

    #for ticker in all_tickers:
    #all_transactions= all_transactions.loc[all_transactions['ticker'] == 'CTHR']
    print(all_transactions['gain_loss'].sum())

    #all_transactions= all_transactions['ticker']=='CTHR'

def printAllRelevantNumbersShortingEdition(dataTickersFromTransactionBuilder):
    last_file = glob(dataTickersFromTransactionBuilder)[-1]
    all_transactions = pd.read_excel(last_file)

    all_tickers = list(all_transactions['ticker'].unique())

    total = all_transactions['gain_loss'].sum()
    print(total)

    print(len(all_tickers))

    print(total / len(all_tickers))

    print((total / len(all_tickers)) / 5000)

    # (AllPrices.loc[date:date]['Close'][0]==0)

    # for ticker in all_tickers:
    # all_transactions= all_transactions.loc[all_transactions['ticker'] == 'CTHR']
    print(all_transactions['gain_loss'].sum())

    # all_transactions= all_transactions['ticker']=='CTHR'

def printAllRelevantNumbersOptionEdition(dataTickersFromTransactionBuilder):
    last_file = glob(dataTickersFromTransactionBuilder)[-1]
    all_transactions = pd.read_excel(last_file)

    all_tickers = list(all_transactions['ticker'].unique())

    total = all_transactions['profitt'].sum()
    print(total)

    print(len(all_tickers))

    print(total / len(all_tickers))

    #print((total / len(all_tickers)) / len(all_tickers))

    # (AllPrices.loc[date:date]['Close'][0]==0)

    # for ticker in all_tickers:
    # all_transactions= all_transactions.loc[all_transactions['ticker'] == 'CTHR']
    #print(all_transactions['gain_loss'].sum())

    # all_transactions= all_transactions['ticker']=='CTHR'


#printAllRelevantNumbers('datatickers123-22015.xlsx')
#printAllRelevantNumbers('datatickers123-22016.xlsx')
#printAllRelevantNumbers('datatickers123-22017.xlsx')
#printAllRelevantNumbers('datatickers123-22018.xlsx')

printAllRelevantNumbersOptionEdition('datatickers123-2.xlsx')
