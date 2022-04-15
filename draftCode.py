import pandas as pd
from glob import glob
from time import strftime, sleep
import numpy as np
from datetime import datetime
from pandas_datareader import data as pdr
from pandas.tseries.offsets import BDay
import yfinance as yf
#yf.pdr_override()
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
#from dash_bootstrap_components import themes as dbc
import plotly.graph_objects as go
from dash import dash_table
#from jupyter_dash import JupyterDash
import webbrowser
from threading import Timer
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from dash.exceptions import PreventUpdate
import dash_table
import plotly.express as px
import logging, sys

from datetime import datetime, timedelta

import pandas as pd
import datetime as dt

import xlsxwriter

import xlrd as xd

#U1346 = df.iloc[lambda x: x.index < 1346]
#O1346 = df.iloc[lambda x: x.index > 1346]


# print('shape til u1300')
# print(U1346)

# print('shape til den nye df')
# print(df.shape)
# print(df)

# sortert = df.index(500)

# print(data)

#print('velger ut de beste aksjene med en cuttoff probabilitet på', cutoff)
#print('resultatet fra utvalg av de beste aksjene - dvs hvor mange prosen er riktige:', len(O1346) / len(df))


# print('orly1')
# print(sannsynligheter.iloc[1240, [0,224] ])

#v2 = 'JHB'
#BuySellOrders = pd.read_excel('datatickers123.xlsx')

#BuySellOrders.to_csv("FromXlscToCSV.csv",
                # index=None,
                # header=True)
#BuySellOrders = pd.DataFrame(pd.read_csv("FromXlscToCSV.csv"))

#print(df['date'])

#df = yf.download(v2, '2019-01-01',datetime.today())

#BuySellOrders = BuySellOrders.loc[BuySellOrders['ticker'] == v2]
#BuySellOrders = BuySellOrders.loc[BuySellOrders['type'] == 'Buy']
#BuySellOrdersDateBought = BuySellOrders['date'].values.tolist()
#BuySellOrdersprice = BuySellOrders['price'].values.tolist()

#print(BuySellOrdersDateBought)


#from datetime import datetime
#excel_date = BuySellOrdersDateBought[0]
#excel_date = 15844032
#print(excel_date)
#excel_date = excel_date.zeros((1, 3))
#datoen = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + excel_date - 2)
#tt = datoen.timetuple()
#print(datoen)#print(tt)


#xd.xldate.xldate_as_tuple(excel_date, 0)

#np.zeros((1, 3))

#BuySellOrdersDateBought = [dt.datetime.strptime(str(date), '"%Y-%m-%d"').date() for date in BuySellOrdersDateBought]


txtDate = '2019-01-01'

txtDate = pd.to_datetime(txtDate, dayfirst=True)

#txtDate=datetime(txtDate)

#test = datetime.strptime(txtDate, '%b %d %Y')

daysSinceStartTime50 = txtDate - timedelta(days=5)

print(daysSinceStartTime50)