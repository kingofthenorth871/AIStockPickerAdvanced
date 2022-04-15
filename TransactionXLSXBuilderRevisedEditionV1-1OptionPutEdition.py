import pandas_datareader as pdr
import datetime as dt

from matplotlib import pyplot as plt

import numpy as np

ticker = "AAPL"
start = dt.datetime(2020, 1, 1)
end = dt.datetime(2020, 12, 31)
data = pdr.get_data_yahoo(ticker, start, end)
data['Log returns'] = np.log(data['Close']/data['Close'].shift())
data['Log returns'].std()
volatility = data['Log returns'].std()*252**.5

#print(volatility)

#str_vol = str(round(volatility, 4)*100)





import time, os, sys  # Standard Python Library
import pandas as pd  # pip install pandas
from yahoofinancials import YahooFinancials  # pip install yahoofinancials
import yfinance as yf
import math
from datetime import datetime, timedelta



def pull_stock_data(tickers):
    """
    Steps:
    1) Create an empty DataFrame
    2) Iterate over tickers, pull data from Yahoo Finance & add data to dictonary "new row"
    3) Append "new row" to DataFrame
    4) Return DataFrame
    """
    if tickers:
        print(f"Iterating over the following tickers: {tickers}")
        df = pd.DataFrame()
        for ticker in tickers:
            print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"Pulling financial data for: {ticker} ...")
            data = YahooFinancials(ticker)
            open_price = data.get_open_price()

            # If no open price can be found, Yahoo Finance will return 'None'
            if open_price is None:
                # If opening price is None, append empty dataframe (row)
                print(f"Ticker: {ticker} not found on Yahoo Finance. Please check")
                df = df.append(pd.Series(dtype=str), ignore_index=True)
            else:
                try:
                    price2019 = yf.download(ticker, '2020-01-01', '2020-01-01')
                    #print('stockPrice')
                    #print(price2019['Close'][0])
                    try:
                        long_name = data.get_stock_quote_type_data()[ticker]["longName"]
                    except (TypeError, KeyError):
                        long_name = None
                    try:
                        yield_rel = data.get_summary_data()[ticker]["yield"]
                    except (TypeError, KeyError):
                        yield_rel = None

                    ticker_currency = data.get_currency()
                    #conversion_rate = get_coversion_rate(ticker_currency)

                    print(ticker)

                    price2019 = yf.download(ticker, '2020-01-01', '2020-01-01')
                    USDBudgetPerStock = 5000

                    stopTradingDate = datetime(2019, 1, 4)
                    stopTrading = False


                    now = datetime.now().strftime('%Y-%m-%d')
                    daysSinceStartTime365 = datetime(2018, 3, 2) - timedelta(days =365)
                    AllPrices = yf.download(ticker, daysSinceStartTime365, now)
                    #DatesToIterateOverAllPrices = AllPrices[0]

                    #df.loc[df['column_name'] == some_value]

                    firstTradingDate = datetime(2018, 3, 2)

                    first = AllPrices.loc[[firstTradingDate]]


                    last = AllPrices.loc[[stopTradingDate]]


                    ## kalkulerer volatiliteten
                    AllPrices['Log returns'] = np.log(AllPrices['Close'] / AllPrices['Close'].shift())
                    AllPrices['Log returns'].std()
                    volatility = AllPrices['Log returns'].std() * 252 ** .5
                    ## kalkulerer volatiliteten


                    #kalkulerer put option prisen
                    N_Days = 252
                    N_Runs = 10000
                    Spot_Price = first['Close'][0]
                    strike = Spot_Price*1
                    volatility = volatility
                    np.random.seed(25)
                    rets = np.random.randn(N_Runs, N_Days) * volatility / np.sqrt(252)
                    #print(rets.shape)
                    traces = np.cumprod(1 + rets, 1) * Spot_Price
                    put = np.mean((strike - traces[:, -1]) * (((traces[:, -1] - strike) < 0)))
                    #kalkulerer put option prisen



                    #DatesToIterateOverAllPrices = DatesToIterateOverAllPrices.index
                    #DatesToIterateOverAllPrices = DatesToIterateOverAllPrices.tolist()

                    buyOrSell = None
                    quantityBought=None
                    gain_loss = None
                    Yield = None
                    BoughtPrice = None

                    quantityHeld = None


                    ##hva må være med i rowen?

                    ## kjøp av opsjonen:

                    ## option type
                    ## ticker
                    ## price
                    ##

                    new_row = {

                        "optiontype" : 'buy Put',
                        "ticker" : ticker,
                        "price" : 1,

                    }
                    df = df.append(new_row, ignore_index=True)

                    ## salg av opsjonen

                    # option type
                    # ticker
                    # profitt
                    #

                    putsellPriceProfitt = 0

                    if (last['Close'][0]<strike):
                        putsellPriceProfitt = ((strike-last['Close'][0])/put)+1

                        putsellPriceProfitt = (((strike-last['Close'][0])*100)/(put*100)) +1

                    new_row = {

                        "optiontype": 'sell Put',
                        "ticker": ticker,
                        "profitt": putsellPriceProfitt,

                    }
                    df = df.append(new_row, ignore_index=True)




                    print(f"Successfully pulled financial data for: {ticker}")

                except Exception as e:
                    # Error Handling
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    #print(AllPrices.loc[date:date]['Close'][0])
                    #print(math.floor(AllPrices.loc[date:date]['Close'][0]))
                    # Append Empty Row
                    df = df.append(pd.Series(dtype=str), ignore_index=True)
        return df
    return pd.DataFrame()


def main():
    print(f"Please wait. The program is running ...")
    #clear_content_in_excel()

    #tickers = ['AAPL', 'GOOGL']
    tickers = pd.read_excel('stockWinnersFromAIStockPicker2.xlsx')
    tickers = tickers['tickers'].values.tolist()

    #del tickers[6]
    print(tickers)

    df = pull_stock_data(tickers)
    df.to_excel('datatickers123-2.xlsx')
    #write_value_to_excel(df)
    print(f"Program ran successfully!")
    #show_msgbox("DONE!")

main()





