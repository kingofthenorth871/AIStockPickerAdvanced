import requests
import json

#c247ca711e240e8c07bce1aa1549214d

#OSE

#income-statement-as-reported

api_url = "https://financialmodelingprep.com/api/v3/quotes/euronext&apikey=c247ca711e240e8c07bce1aa1549214d"
api_url = "https://financialmodelingprep.com/api/v3/cash-flow-statement/BSP.OL?limit=10&apikey=c247ca711e240e8c07bce1aa1549214d"

#api_url = "https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/BSP.OL?apikey=c247ca711e240e8c07bce1aa1549214d"

#api_url = "https://financialmodelingprep.com/api/v4/financial-reports-json?symbol=BSP.OL&year=2020&period=FY&apikey=c247ca711e240e8c07bce1aa1549214d"

api_url =  "https://financialmodelingprep.com/api/v3/ratios-ttm/BSP.OL?apikey=c247ca711e240e8c07bce1aa1549214d"

api_url = "https://financialmodelingprep.com/api/v3/enterprise-values/BSP.OL?limit=40&apikey=c247ca711e240e8c07bce1aa1549214d"

api_url = "https://financialmodelingprep.com/api/v3/key-metrics-ttm/BSP.OL?limit=40&apikey=c247ca711e240e8c07bce1aa1549214d"

api_url = "https://financialmodelingprep.com/api/v3/rating/ANDF.OL?apikey=c247ca711e240e8c07bce1aa1549214d"

api_url = "https://financialmodelingprep.com/api/v4/financial-reports-json?symbol=AVGO&year=2018&period=FY&apikey=c247ca711e240e8c07bce1aa1549214d"

api_url = 'https://financialmodelingprep.com/api/v3/financials/income-statement/BSP.OL?&apikey=c247ca711e240e8c07bce1aa1549214d'

response = requests.get(api_url)

myJson = response.json()

with open('data.txt', 'w') as outfile:
    json.dump(myJson, outfile)

print(myJson)