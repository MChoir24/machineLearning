import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def linearRegression(x, y):
    xt = np.transpose(x)
    xinv = np.linalg.inv(np.dot(xt, x))
    temp = np.dot(xinv, xt)
    w = np.dot(temp, y)
    return w

def predict(w, data):
    temp = w[1:] * data
    temp1 = np.concatenate((w[0],temp), axis=None)
    return sum(temp1)

def predicts(w, datas):
    jumlahBaris = datas.shape[0]
    result = np.zeros(jumlahBaris)
    for i in range(jumlahBaris):
        result[i] = predict(w, datas[i])
    return result

def R2(target, predict):
    sstot = sum((target - np.average(target)) ** 2)
    ssres = sum((target - predict) ** 2)
    result = 1 - (ssres/sstot)
    return(result)

def splitData(dataset, p=100):
    if p >= 100:
        dataTrain = dataset.data
        target = dataset.target
        dataTest = dataset.data
        targetTest = dataset.target
    else:
        jumlahDataTrain = int((p * dataset.data.shape[0]) / 100)
        # data train
        dataTrain = dataset.data[:jumlahDataTrain]
        target = dataset.target[:jumlahDataTrain]

        # data test
        dataTest = dataset.data[jumlahDataTrain:]
        targeTest = dataset.target[jumlahDataTrain:]

    return dataTrain, target, dataTest, targeTest

# load data
dataset = datasets.load_diabetes()

# split Data
dataTrain, target, dataTest, targeTest = splitData(dataset, p=80)

# menambah matrix ones pada data train
ones = np.ones((dataTrain.shape[0], 1))
dataTrain1 = np.concatenate((ones, dataTrain), axis=1)

# manual
w = linearRegression(dataTrain1, target)
predict = predicts(w, dataTest)

# scikitLearn
reg = LinearRegression().fit(dataTrain, target)
predictSK = reg.predict(dataTest)

# score manual
print(r2_score(targeTest, predict))
# score scikir
print(r2_score(targeTest, predictSK))
