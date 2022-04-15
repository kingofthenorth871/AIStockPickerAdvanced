from enum import Enum  # Standard Python Library
import time, os, sys  # Standard Python Library
#import xlwings as xw  # pip install xlwings
import pandas as pd  # pip install pandas
from yahoofinancials import YahooFinancials  # pip install yahoofinancials
import yfinance as yf

import math
from datetime import datetime, timedelta

# ==============================
# Purpose:
# Returning stock, cryptocurrency, forex, mutual fund, commodity futures, ETF,
# and US Treasury financial data from Yahoo Finance & export it MS EXCEL
#
# Hints:
# In case you want to adjust/change/add more information to the worksheet,
# make sure to do your respective adjustments in the following:
#   a) class Column(Enum): change/add the column-number/name
#   b) adjust the dictonary "new_row" in the function "pull_stock_data()"
# ==============================

print(
    """
==============================
Dividend & Portfolio Overview
==============================
"""
)


class Column(Enum):
    """ Column Name Translation from Excel, 1 = Column A, 2 = Column B, ... """

    long_name = 1
    ticker = 2
    current_price = 5
    currency = 6
    conversion_rate = 7
    open_price = 8
    daily_low = 9
    daily_high = 10
    yearly_low = 11
    yearly_high = 12
    fifty_day_moving_avg = 13
    twohundred_day_moving_avg = 14
    payout_ratio = 19
    exdividend_date = 20
    yield_rel = 21
    dividend_rate = 22


def timestamp():
    t = time.localtime()
    timestamp = time.strftime("%b-%d-%Y_%H:%M:%S", t)
    return timestamp


#def clear_content_in_excel():
    #"""Clear the old contents in Excel"""
    #if LAST_ROW > START_ROW:
       # print(f"Clear Contents from row {START_ROW} to {LAST_ROW}")
        #for data in Column:
          #  if not data.name == "ticker":
                #sht.range((START_ROW, data.value), (LAST_ROW, data.value)).options(
                #    expand="down"
               # ).clear_contents()
       # return None


#def convert_to_target_currency(yf_retrieve_data, conversion_rate):
    #"""If value is not available on Yahoo finance, it will return None"""
    #if yf_retrieve_data is None:
     #   return None
   # return yf_retrieve_data * conversion_rate


#def get_coversion_rate(ticker_currency):
    """
    Calculate the coversion rate between
    ticker currency & desired output currency (TARGET_CURRENCY)
    Return: conversion rate
    """
   # if TARGET_CURRENCY == "TICKER CURRENCY":
       # print(f"Display values in {ticker_currency}")
       # conversion_rate = 1
       # return conversion_rate
   # conversion_rate = YahooFinancials(
      #  f"{ticker_currency}{TARGET_CURRENCY}=X"
   # ).get_current_price()
   # print(
    #    f"Conversion Rate from {ticker_currency} to {TARGET_CURRENCY}: {conversion_rate}"
    #)
    #return conversion_rate

def pull_stock_dataProfitt(ticker, numberOfStocks, datefrom, dateto):
    data1 = YahooFinancials(ticker)
    data1 = yf.download(ticker, datefrom, datefrom)
    data1 = pd.DataFrame(data1)
    data2 = YahooFinancials(ticker)
    data2 = yf.download(ticker, dateto, dateto)
    data2 = pd.DataFrame(data2)

    profit = (data2['Close'].iloc[0] - data1['Close'].iloc[0]) * numberOfStocks
    return profit

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


                    now = datetime.now().strftime('%Y-%m-%d')
                    twentitwentione = datetime(2016, 1, 1)
                    daysSinceStartTime200 = datetime(2016, 1, 1) - timedelta(days=200)
                    AllPrices = yf.download(ticker, daysSinceStartTime200, now)

                    DatesToIterateOverAllPrices = AllPrices.loc[twentitwentione:now]
                    DatesToIterateOverAllPrices = DatesToIterateOverAllPrices.index
                    DatesToIterateOverAllPrices = DatesToIterateOverAllPrices.tolist()

                    buyOrSell = None
                    quantityBought=None
                    gain_loss = None
                    Yield = None
                    BoughtPrice = None

                    quantityHeld = None


                    #price = AllPrices.loc[date:date]['Close'][0]
                    #quantity = math.floor(USDBudgetPerStock / price)

                    stopTradingDate = datetime(2017, 2, 2)
                    stopTrading = False



                    for date in DatesToIterateOverAllPrices:



                        if(stopTradingDate ==date):
                            print('stop trading!')
                            stopTrading = True;

                        if(AllPrices.loc[date:date]['Close'][0]==0):
                            date = date + timedelta(days=1)
                        if (AllPrices.loc[date:date]['Close'][0] == 0):
                            date = date + timedelta(days=1)

                        price = AllPrices.loc[date:date]['Close'][0]

                        if(quantityBought==None):
                            quantity = math.floor(USDBudgetPerStock / price)
                        else:
                            quantity=quantityBought

                        daysSinceStartTime50 = date - timedelta(days=60)   #5
                        fiftyDayMovingAverage = AllPrices.loc[daysSinceStartTime50:date]
                        fiftyDayMovingAverage = fiftyDayMovingAverage['Close'].mean()
                        #print('fifty day moving average: ')
                        #print(fiftyDayMovingAverage)

                        daysSinceStartTime200 = date - timedelta(days=15)      #20
                        twoHoundredDayMovingAverage = AllPrices.loc[daysSinceStartTime200:date]
                        twoHoundredDayMovingAverage = twoHoundredDayMovingAverage['Close'].mean()
                        #print('two houndred day moving average: ')
                        #print(twoHoundredDayMovingAverage)

                        ##fiftyDayMovingAverage<twoHoundredDayMovingAverage and buyOrSell!='Sell' and buyOrSell!=None or stopTrading==True and buyOrSell!='Sell'
                        if (buyOrSell!='Sell' and stopTrading!=True):
                            print('Sell')
                            print('50 day moving average')
                            print(fiftyDayMovingAverage)
                            print('200 day moving average')
                            print(twoHoundredDayMovingAverage)
                            print('Sell price')
                            print(price)

                            buyOrSell = 'Sell'

                            new_row = {
                                "date": date,
                                "type": 'Sell',
                                "ticker": ticker,
                                "quantity": quantityHeld,
                                "price": price,
                                "fees": 0,
                                "transact_val": price*quantity,
                                "last_occurence": 0,
                                "cashflow": price*quantity,
                                "prev_units": 0,
                                "cml_units": USDBudgetPerStock / (AllPrices.loc[date:date]['Close'][0]),
                                "prev_cost": 0,
                                "cml_cost": 0,
                                "cost_transact": 0,##BoughtPrice*quantity,
                                "cost_unit": BoughtPrice,
                                "gain_loss": 0,##(price*quantity)-(BoughtPrice*quantity),
                                "yield":  0, #(price*quantity)/(BoughtPrice*quantity),
                                "avg_price": AllPrices.loc[date:date]['Close'][0],

                            }
                            USDBudgetPerStock = price*quantity
                            df = df.append(new_row, ignore_index=True)

                        ##fiftyDayMovingAverage>twoHoundredDayMovingAverage and buyOrSell!='Buy' and stopTrading==False
                        if (stopTrading==True and buyOrSell!='Buy'):
                            print('Buy')
                            print('50 day moving average')
                            print(fiftyDayMovingAverage)
                            print('200 day moving average')
                            print(twoHoundredDayMovingAverage)
                            print('Buy price')
                            print(price)
                            buyOrSell = 'Buy'
                            quantityBought = quantity
                            BoughtPrice = price

                            quantity = USDBudgetPerStock / price

                            new_row = {
                                "date": date,
                                "type": 'Buy',
                                "ticker": ticker,
                                "quantity": USDBudgetPerStock/price,
                                "price": AllPrices.loc[date:date]['Close'][0],
                                "fees": 0,
                                "transact_val": price*quantity,
                                "last_occurence": 0,
                                "cashflow": -(price*quantity),
                                "prev_units": 0,
                                "cml_units": USDBudgetPerStock / (AllPrices.loc[date:date]['Close'][0]),
                                "prev_cost": 0,
                                "cml_cost": price*quantity,
                                "cost_transact": 0,
                                "cost_unit": 0,
                                "gain_loss": 0,
                                "yield": 0,
                                "avg_price": AllPrices.loc[date:date]['Close'][0],
                            }
                            quantityHeld = USDBudgetPerStock/price
                            df = df.append(new_row, ignore_index=True)


                        #print('datoen:')
                        #print(date)



                    print(f"Successfully pulled financial data for: {ticker}")

                except Exception as e:
                    # Error Handling
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(AllPrices.loc[date:date]['Close'][0])
                    print(math.floor(AllPrices.loc[date:date]['Close'][0]))
                    # Append Empty Row
                    df = df.append(pd.Series(dtype=str), ignore_index=True)
        return df
    return pd.DataFrame()


#def write_value_to_excel(df):
    #if not df.empty:
        #print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print(f"Writing data to Excel...")
        #options = dict(index=False, header=False)
       # for data in Column:
           # if not data.name == "ticker":
              #  sht.range(START_ROW, data.value).options(**options).value = df[
               #     data.name
               # ]
       # return None


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
# --- GET VALUES FROM EXCEL
# xw.Book.caller() References the calling book
# when the Python function is called from Excel via RunPython.





#wb = xw.Book.caller()
#sht = wb.sheets("Portfolio")
#show_msgbox = wb.macro("modMsgBox.ShowMsgBox")
#TARGET_CURRENCY = sht.range("TARGET_CURRENCY").value
#START_ROW = sht.range("TICKER").row + 1  # Plus one row after the heading
#LAST_ROW = sht.range(sht.cells.last_cell.row, Column.ticker.value).end("up").row
#sht.range("TIMESTAMP").value = timestamp()
#tickers = (
   # sht.range(START_ROW, Column.ticker.value).options(expand="down", numbers=str).value
#)