import requests
import json
api_url = url = 'https://financialmodelingprep.com/api/v3/stock/list?apikey=c247ca711e240e8c07bce1aa1549214d'
response = requests.get(api_url)

url0 = 'https://financialmodelingprep.com/api/v3/financials/income-statement/' + tickers_found[t] + accessCode
url1 = 'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/' + tickers_found[t] + accessCode
url2 = 'https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/' + tickers_found[t] + accessCode
url3 = 'https://financialmodelingprep.com/api/v3/financial-ratios/' + tickers_found[t] + accessCode
url4 = 'https://financialmodelingprep.com/api/v3/company-key-metrics/' + tickers_found[t] + accessCode
url5 = 'https://financialmodelingprep.com/api/v3/financial-statement-growth/' + tickers_found[t] + accessCode


myJson = response.json()

print(myJson)

