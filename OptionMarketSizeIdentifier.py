from yahoo_fin import options
import pandas as pd

stock = 'AAPL'
pd.set_option('display.max_columns', None)
expirationDate = 'January 20, 2023'
type = 'calls'
cutOffPoint = 0.75

#print(options.get_expiration_dates(stock))

chain = options.get_options_chain(stock, expirationDate)
chain = chain[type]

chain['bidAskSpread'] = chain['Bid']/chain['Ask']
print(chain['bidAskSpread'].mean())

tickers = pd.read_excel('stockWinnersFromAIStockPicker2.xlsx')
tickers = tickers['tickers'].values.tolist()

allTickers = []

for ticker in tickers:
    try:
        stock = ticker
        print(options.get_expiration_dates(stock))
        chain = options.get_options_chain(stock, expirationDate)
        chain = chain[type]
        chain['bidAskSpread'] = chain['Bid'] / chain['Ask']
        print(stock)
        print(chain['bidAskSpread'].mean())
        if (ticker not in allTickers and chain['bidAskSpread'].mean()>cutOffPoint):
            allTickers.append(ticker)
    except:
        print(stock)
        print('no option market for stock')

print(allTickers)