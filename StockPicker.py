from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from heapq import nlargest, nsmallest


def createEqualAmountOfClassesHardcoded(data):
    klasse1 = data[data['Class'] == 1]
    klasse2 = data[data['Class'] == 0]
    klasse1 = klasse1.iloc[0:1346, :]
    frames = [klasse1, klasse2]
    result = pd.concat(frames)
    return result

def fillInMissingValues(X, data):
    X = X.fillna(data.mean())
    return X

def scaleData(X):

    Xoriginal = X
    X = MinMaxScaler().fit_transform(X)

    X = pd.DataFrame(X, columns=Xoriginal.columns)

    #print('X dataframe etter skalering1:')
    #print(X)

    return X

def getListOfSelectedFeatureNames(X_selected_df, originalValuesAsDF):
    i = 0
    selectedStocksAttributes = []
    while i < len(X_selected_df):
        if (X_selected_df[i] == True):
            selectedStocksAttributes.insert(i, originalValuesAsDF[i])
        i = i + 1
    return selectedStocksAttributes


def selectBestFeatures(X, y):

    Xoriginal = X
    #print('X original')
    #print(Xoriginal)

    m = SelectFromModel(svm.SVC(max_iter=100000, C=1, kernel='linear', probability=True))
    m.fit(X, y)

    X_selected_df = m.get_support()

    X = m.transform(X)

    originalValuesAsDF = list(Xoriginal.columns.values.tolist())

    selectedStocksAttributes = getListOfSelectedFeatureNames(X_selected_df, originalValuesAsDF)

    print('selected stock attributes: ')
    print(selectedStocksAttributes)

    return X, selectedStocksAttributes

def prepareData(X, data, y):
    X = fillInMissingValues(X, data)
    X = scaleData(X)
    X, selectedStocksAttributes = selectBestFeatures(X, y)

    return X, selectedStocksAttributes

def trainClassifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classifier = svm.SVC(max_iter=100000, C=1, kernel='linear', probability=True)
    # classifier = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(10, 10, 10, 5), random_state=1)
    classifier.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, classifier

def winningAndLoosingStocks(df, dataOld):
    listWithIndex = df.index.values.tolist()
    winners = 0
    loosers = 0
    for entry in listWithIndex:
        #print(len(dataOld.columns))
        #print(dataOld.iloc[entry, len(dataOld.columns)-1])
        stock = dataOld.iloc[entry, len(dataOld.columns)-1]
        #print(classificationCriterium)

        #print('printer stock')
        #print(stock)
        #if stock['class20'] == 1:
        if stock == 1:
            winners = winners + 1
        else:
            loosers = loosers + 1
    return winners, loosers, listWithIndex

def printOutWinnersAndLoosers(winners, loosers, listWithIndex, dataOld):
    print('Percentage of correct winners:')
    print(winners / (winners + loosers))
    print('number of stocks:')
    print(len(listWithIndex))
    print('stock names:')
    winnerStockDataFrame = pd.DataFrame(dataOld.iloc[listWithIndex, [0, len(dataOld.columns)-1]])
    print(winnerStockDataFrame)
    winnerStockDataFrame = winnerStockDataFrame.rename(columns={'Unnamed: 0': 'tickers'})
    winnerStockDataFrame.to_excel('stockWinnersFromAIStockPicker.xlsx')
    print(dataOld.iloc[listWithIndex, [0, len(dataOld.columns)-1]])

def pickOutWinnersBasedOnProbabilityFromClassifier(probability, Chosenclassifier, Xvalue, data):
    dataOld = data # pd.read_csv("SelfMadeStockDataset.csv")
    probabilities = Chosenclassifier.predict_proba(Xvalue)
    df = pd.DataFrame(probabilities, columns=['Column_A', 'Column_B'])
    df = df.loc[(df['Column_B'] >= probability)]
    winners, loosers, listWithIndex=winningAndLoosingStocks(df, dataOld)
    printOutWinnersAndLoosers(winners, loosers, listWithIndex, dataOld)

def printClassifierScores(classifier, X, y, y_test, Y_pred):
    print('score of classifier:')
    print(classifier.score(X, y))
    F1 = f1_score(y_test, Y_pred, average='micro')
    acc = accuracy_score(y_test, Y_pred)
    print('accuracy score:')
    print(acc)
    print('f1 score: ')
    print(F1)

def trainClassifierAndPickOutBestStocks(probability, Xval, yval, data):
    print("printer ut X før")
    print(Xval)

    X, selectedStocksAttributes = prepareData(Xval, data, yval)
    X_train, X_test, y_train, y_test, classifier = trainClassifier(X, yval)
    Y_pred = classifier.predict(X_test)
    pickOutWinnersBasedOnProbabilityFromClassifier(probability, classifier, X, data)
    printClassifierScores(classifier, X, y, y_test, Y_pred)


    ## new dataset on old classifier

    data2019or = pd.read_csv("SelfMadeStockDatasetCleanedInStockPicker2019.csv")

    data2019or = data2019or.drop('Unnamed: 0', 1)

    data2019or = data2019or.rename(columns={'Unnamed: 0.1': 'Unnamed: 0'})

    data2018or = pd.read_csv("SelfMadeStockDatasetCleanedInStockPicker2018.csv")

    #mod_data.drop('index', axis=1, inplace=True)

    #data2019or = data2019or.drop('Dividend Yield',1)

    cols = [c for c in data2019or.columns if c in selectedStocksAttributes]

    data2019cols = data2019or[cols]

    data2019cols = data2019cols.fillna(data2019cols.mean())

    data2019cols = MinMaxScaler().fit_transform(data2019cols)

    print("prediction: ")
    print(classifier.predict(data2019cols))

    pickOutWinnersBasedOnProbabilityFromClassifier(0.62, classifier, data2019cols, data2019or)

    ## new dataset on old classifier


    print("printer ut X etter")
    print(X)

    print('printer coef:')
    print(classifier.coef_)
    listOfcoefficients = classifier.coef_
    print('item nr 2 i listen:')
    specificCoefficientsList = listOfcoefficients[0]
    print(specificCoefficientsList[1])
    print(len(specificCoefficientsList))

    mostImportantFactors = nlargest(10, range(len(specificCoefficientsList)), key=lambda idx: specificCoefficientsList[idx])

    i=0
    while i<len(mostImportantFactors):
        print(specificCoefficientsList[mostImportantFactors[i]])
        print(selectedStocksAttributes[mostImportantFactors[i]])
        i = i + 1

    mostImportantNegativeFactors = nsmallest(10, range(len(specificCoefficientsList)), key=lambda idx: specificCoefficientsList[idx])

    i = 0
    while i < len(mostImportantNegativeFactors):
        print(specificCoefficientsList[mostImportantNegativeFactors[i]])
        print(selectedStocksAttributes[mostImportantNegativeFactors[i]])
        i = i + 1

data = pd.read_csv("SelfMadeStockDataset2018.csv")

print('printer column lengden')
print(len(data.columns))
print(len(data))

mod_data = data.dropna(thresh=45)
mod_data['priceVar1yr'] = mod_data['priceVar1yr'].map(lambda x: x.lstrip('[').rstrip(']'))
mod_data['priceVar1yr'] = mod_data['priceVar1yr'].astype(float).round(0)
mod_data['priceVar1yr'] = mod_data['priceVar1yr'].astype(int)




mod_data.drop('date', 1)
mod_data.drop('priceVar1yr',1)
mod_data.drop('Unnamed: 0', 1)
print('mod_dataLength')
print(len(mod_data))
mod_data = mod_data.fillna(mod_data.mean())
mod_data=mod_data.round(2)



# Remove columns that have more than 20 0-values
mod_data = mod_data.loc[:, mod_data.isin([0]).sum() <= 2500]

# Remove columns that have more than 15 nan-values
mod_data = mod_data.loc[:, mod_data.isna().sum() <= 2500]

# Fill remaining nan-values with column mean value
mod_data.iloc[:, 2:225] = mod_data.iloc[:, 2:225].apply(lambda x: x.fillna(x.mean()))

conditions = [
    mod_data['priceVar1yr'] > 40 , mod_data['priceVar1yr'] < 40
]

choices = [1,0]
mod_data['class'] = np.select(conditions, choices, default=0)

print(mod_data)
#mod_data = mod_data.reset_index()

mod_data.to_csv('SelfMadeStockDatasetCleanedInStockPicker2018.csv')

classificationCriterium = 'class'
X = mod_data.iloc[:, 2:223]

#mod_data.drop('index', axis=1, inplace=True)

#mod_data.drop('Unnamed', axis=1, inplace=True)


y = mod_data[classificationCriterium]
X = mod_data.iloc[:, 2:len(mod_data.columns)-2]

print('printer X')
print(X)

print('printer Y')
print(y)





#print('printer X')
#print(X)




trainClassifierAndPickOutBestStocks(0.60, X, y, mod_data)






