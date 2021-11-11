import requests
import json
import time

usedTicker= [ "CHK", "DIS", "AMD","AAPL","XOM","AVGO", "ORCL", "NKE", "PG", "GSM", "BBD", "MSCI", "RDN", "KO", "AABA", "SRC", "COP"]
#usedTicker= ["AVGO"]

i=1

for symbol in usedTicker:

    time.sleep(2)

    print(symbol)

    api_url = "https://financialmodelingprep.com/api/v4/financial-reports-json?symbol=symbolTicker&year=2018&period=FY&apikey=c247ca711e240e8c07bce1aa1549214d"

    api_url = api_url.replace("symbolTicker", symbol)

    print(i)
    i= i+1

    print(api_url)

    response = requests.get(api_url)
    myJson = response.json()

    #Revenue = myJson['Consolidated Statements of Oper'][2]['Net revenue'][16]
    #print(Revenue)

    try:
        print('round 1')
        Revenue = myJson['Consolidated Statements of Oper'][2]['Net revenue'][9]

    except:
        pass

        try:
            print('round 2')
            Revenue = myJson['CONSOLIDATED STATEMENTS OF CASH'][3]['Net income'][0]
        except:
            pass

            try:
                print('round 3')
                Revenue = myJson['Consolidated Statement Of Incom'][5]['Total revenues and other income'][1]
            except:
                pass

                try:
                    print('round 4')
                    Revenue = myJson['Consolidated Statements of Oper'][2]['Net revenue'][16]
                except:
                    print('something went wrong')



    #Consolidated Statements of Oper

    if Revenue==None:
        Revenue = myJson['Consolidated Statements of Oper'][2]['Net revenue'][16]

    print(Revenue)