import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


diabetes = datasets.load_diabetes()

def buildMatrix(x):
    [numOfData,feature]=x.shape
    newX=np.ones((numOfData,feature+1))
    newX[:,1:]=x
    return newX

def stochasticGD(x,y,epoch,etha):
    [numOfData,feature]=x.shape
    weight=np.zeros((epoch,feature+1)) #+1 for bias
    tempWeight=np.random.randn(1,feature+1)
    tempData=buildMatrix(x)
    for i in range(epoch):
        for j in range(numOfData):
            prediction=tempWeight.dot(tempData[j])
            error=prediction - y[j]
            tempWeight=tempWeight-etha*error*tempData[j]
        weight[i]=tempWeight
    return weight

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


dataTrain, target, dataTest, targeTest = splitData(diabetes, p=80)

w=stochasticGD(dataTrain,target,100,0.02)
# print(dataTest)
predict = predict(w[len(w)-1], dataTrain[25])
print(predict)
# print(r2_score(targeTest, predict))
# for i in range(10,100,1):
#     pass
