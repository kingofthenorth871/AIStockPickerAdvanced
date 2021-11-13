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

    m = SelectFromModel(svm.SVC(max_iter=100000, C=30, kernel='linear', probability=True))
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
    classifier = svm.SVC(max_iter=100000, C=30, kernel='linear', probability=True)
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

data = pd.read_csv("SelfMadeStockDataset.csv")

#print(data.iloc[:, 223:224])

data = data.drop('class10', 1)
data = data.drop('class20', 1)
data = data.drop('class80', 1)

print('printer column lengden')
print(len(data.columns))

test = data.iloc[:, 181:184]
print('test')
print(test)

classificationCriterium = 'class40'
X = data.iloc[:, 1:181]

y = data[classificationCriterium]
X = data.iloc[:, 1:len(data.columns)-1]

print('printer X')
print(X)




trainClassifierAndPickOutBestStocks(0.70, X, y, data)






