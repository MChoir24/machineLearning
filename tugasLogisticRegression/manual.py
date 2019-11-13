import numpy as np
import loadCSV
from math import exp
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def fx(x, w):
    temp = w * x
    temp = sum(temp)*(-1)
    result = 1 / (1 + (exp(temp)))
    return result


def training(x, y, alfa, epho):
    x = buildMatrix(x)
    numberOfData = x.shape
    w = np.zeros(numberOfData[1])
    weights = np.zeros((epho, numberOfData[1]))
    for e in range(epho):
        for i in range(numberOfData[0]):
            temp = alfa * (y[i] - fx(x[i], w))
            temp1 = 1 - fx(x[i], w)
            w = w + (temp * fx(x[i], w) * temp1) * x[i]
        weights[e] = w
    return weights


def predict(x, w):
    x = buildMatrix(x)
    result = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        result[i] = fx(x[i], w)
    return result


def accurate(target, predict):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    matConfusi = np.zeros((2,2))
    for i in range(len(target)):
        if (target[i] == 1) and (target[i] == predict[i]):
            tp+=1
        elif (target[i] == 0) and (target[i] == predict[i]):
            tn+=1
        elif (target[i] == 0) and (target[i] != predict[i]):
            fp+=1
        elif (target[i] == 1) and (target[i] != predict[i]):
            fn+=1
    matConfusi[0][0] = tp
    matConfusi[0][1] = fp
    matConfusi[1][0] = fn
    matConfusi[1][1] = tn

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall, matConfusi


def buildMatrix(x):
    # add matrix ones
    numberOfData = x.shape
    ones = np.ones((numberOfData[0], numberOfData[1]+1))
    ones[:, 1:] = x
    return ones


# load data from csv file
datasets = loadCSV.LoadCSV('heart.csv', ',')
# print(datasets)
data = datasets[:, :-1]
target = datasets[:, -1]

# split data training and data test
data_training, data_test, target_training, target_test = train_test_split(data, target, test_size=0.30, random_state=0)

# training model manual
w = training(data_training, target_training, 0.3, 10)

# training model scikit
classifier = LogisticRegression()
classifier.fit(data_training,target_training)
pred=classifier.predict(data[147:157])
# print(pred)

print('\npredict Akurasi manual dengan model manual')
predict = predict(data_test[:10], w[len(w)-1])
print("Akurasi \nPresisi: %s \nRecall: %s \nMatrik Confusion: \n%s "%accurate(target_test[:10], predict))
print("Akurasi \nPresisi: %s \nRecall: %s \nMatrik Confusion: \n%s "%accurate(target_test[:10], predict))
print(target_test[:10])
print(predict)
# print('\npredict Akurasi manual dengan model scikit')
# print("Akurasi \nPresisi: %s \nRecall: %s \nMatrik Confusion: \n%s "%accurate(target[70:80], pred))
# print("Akurasi \nPresisi: %s \nRecall: %s \nMatrik Confusion: \n%s "%accurate(target[15  :25], pred))
#
# print('\npredict Akurasi scikit dengan model manual')
# print('presisi',precision_score(target[70:80], predict))
# print('presisi',precision_score(target[15:25], predict))
# print('recall',recall_score(target[70:80], predict))
#
# print('\npredict Akurasi scikit dengan model scikit')
# print('presisi',precision_score(target[70:80], pred))
# print('recall',recall_score(target[70:80], pred))
