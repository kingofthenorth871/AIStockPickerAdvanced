import requests
import json
api_url = "https://financialmodelingprep.com/api/v4/financial-reports-json?symbol=AAPL&year=2018&period=FY&apikey=c247ca711e240e8c07bce1aa1549214d"
response = requests.get(api_url)

myJson = response.json()

jsonResponseOriginal = myJson

Revenue = myJson['CONSOLIDATED STATEMENTS OF CASH'][3]['Net income'][0]
#Revenue Growth
CostOfRevenue = myJson['CONSOLIDATED STATEMENTS OF OPER'][3]['Cost of sales'][0]
GrossProfit = myJson['CONSOLIDATED STATEMENTS OF OPER'][4]['Gross margin'][0]
RAndDExpenses = myJson['CONSOLIDATED STATEMENTS OF OPER'][5]['Research and development'][0]
SGandAExpense = myJson['CONSOLIDATED STATEMENTS OF OPER'][6]['Selling, general and administrative'][0]
TotalOperatingExpenses = myJson['CONSOLIDATED STATEMENTS OF OPER'][7]['Total operating expenses'][0]
OperatingIncome = myJson['CONSOLIDATED STATEMENTS OF OPER'][8]['Operating income'][0]

#myJson = myJson[3]

print(Revenue)
print(CostOfRevenue)
print(GrossProfit)
print(RAndDExpenses)
print(SGandAExpense)
print(TotalOperatingExpenses)
print(OperatingIncome)

with open('data.txt', 'w') as outfile:
    json.dump(Revenue, outfile)

#c247ca711e240e8c07bce1aa1549214d

#https://financialmodelingprep.com/api/v3/income-statement/AAPL?limit=120&apikey=YOUR_API_KEY'

#api_url = "https://financialmodelingprep.com/api/v3/quote/AAPL?apikey=c247ca711e240e8c07bce1aa1549214d"
response = requests.get(api_url)

myJson = response.json()

print(myJson)
