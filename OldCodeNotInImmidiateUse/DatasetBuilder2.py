import requests
import json
api_url = "https://financialmodelingprep.com/api/v4/financial-reports-json?symbol=AMD&year=2018&period=FY&apikey=c247ca711e240e8c07bce1aa1549214d"
response = requests.get(api_url)

myJson = response.json()

jsonResponseOriginal = myJson

Revenue = myJson['Consolidated Statements of Oper'][2]['Net revenue'][9]
#Revenue Growth
CostOfRevenue = myJson['Consolidated Statements of Oper'][3]['Cost of sales'][9]
GrossProfit = myJson['Consolidated Statements of Oper'][4]['Gross margin'][9]
RAndDExpenses = myJson['Consolidated Statements of Oper'][5]['Research and development'][9]
SGandAExpense = myJson['Consolidated Statements of Oper'][6]['Marketing, general and administrative'][9]
TotalOperatingExpenses = myJson['Consolidated Statements of Oper'][6]['Marketing, general and administrative'][9]

#myJson = myJson[3]

#with open('data.txt', 'w') as outfile:
    #json.dump(Revenue, outfile)

#c247ca711e240e8c07bce1aa1549214d

#https://financialmodelingprep.com/api/v3/income-statement/AAPL?limit=120&apikey=YOUR_API_KEY'

#api_url = "https://financialmodelingprep.com/api/v3/quote/AAPL?apikey=c247ca711e240e8c07bce1aa1549214d"
response = requests.get(api_url)

myJson = response.json()

print(Revenue)
print(CostOfRevenue)
print(GrossProfit)
print(RAndDExpenses)
print(SGandAExpense)

api_url = "https://financialmodelingprep.com/api/v4/financial-reports-json?symbol=NOK&year=2018&period=FY&apikey=c247ca711e240e8c07bce1aa1549214d"
response = requests.get(api_url)

myJson = response.json()
#Revenue = myJson['Consolidated Statements of Oper'][2]['Net revenue'][9]

with open('data.txt', 'w') as outfile:
    json.dump(myJson, outfile)

print(myJson)