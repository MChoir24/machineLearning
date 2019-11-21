import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class ArtificialNeuralNetwork:
    def __init__(self, dataset, target, epoch, lr=0.3, hidden_layer=[1]):
        self.epoch = epoch
        self.hidden_layer = hidden_layer
        self.lr = lr
        self.dataset = dataset
        self.datasetones = self.__builtMatrik()
        self.target = target
        self.weights = self.__weightInit()
        # self.weights = [[[0.1,0.3,0.4],[0.2,0.4,0.15],[0.1,0.1,0.25],[0.15,0.3,0.4],[0.2,0.2,0.5]]
        #                 ,[[0.3,0.2],[0.05,0.1],[0.1,0.3],[0.4,0.4]]]
        self.sigmoid = []

    def __weightInit(self):
        numOfInput = self.datasetones.shape[1]
        weight = []
        for i in range(len(self.hidden_layer)):
            weight.append(np.random.randint(-10,10, (numOfInput, self.hidden_layer[i])) / 10)
            # print (np.random.randint(-10,10, (numOfInput, self.hidden_layer[i])) / 10)
            numOfInput = weight[i].shape[1] + 1
        weight.append(np.random.randint(-10,10, (numOfInput, self.target.shape[1])) / 10)
        return weight

    def __builtMatrik(self, data=None):
        if type(data) == np.ndarray or type(data) == list:
            ones = np.ones((data.shape[0], data.shape[1]+1))
            ones[:, 1:] = data
            return ones
        else:
            ones = np.ones((self.dataset.shape[0], self.dataset.shape[1]+1))
            ones[:, 1:] = self.dataset
            return ones

    def __f(self, n):
        return n * (1 - n)

    def __forwardPass(self, ind):
        self.sigmoid = []
        layer = self.datasetones[ind]
        for i in range(len(self.weights)-1):
            sigmoid = layer.dot(self.weights[i])
            # print(sigmoid.shape)
            layer = 1 / (1 + np.exp(-sigmoid)) # sigmoid
            ones = np.ones(layer.shape[0]+1) # add matrix one
            ones[1:] = layer
            layer = ones
            self.sigmoid.append(layer)

        sigmoid = layer.dot(self.weights[i+1])
        layer = 1 / (1 + np.exp(-sigmoid)) # sigmoid
        self.sigmoid.append(layer)
        return layer


    def __backwardPass(self, output, ind):
        error =  (self.target[ind] - output)
        fok = self.__f(output)
        sigmaO = error * fok
        sigmaO = sigmaO.reshape((sigmaO.shape[0], 1))

        for i in range(len(self.sigmoid)-2, -1, -1):
            sigmoid = self.sigmoid[i].reshape((1, len(self.sigmoid[i])))
            vjk = -self.lr * (-sigmaO.dot(sigmoid)).transpose()
            self.weights[i+1] += vjk
            sigmayj = self.weights[i+1][1:].dot(sigmaO) * (self.__f(sigmoid.transpose()[1:]))
            sigmaO = sigmayj
            # sigmayj = self.weights[i+1][1:].dot(sigmaO)

        sigmoid = self.datasetones[i].reshape((1, len(self.datasetones[i])))
        vjk = -self.lr * (-sigmaO.dot(sigmoid)).transpose()
        self.weights[0] += vjk


    def train(self):
        for i in range(self.epoch):
            for j in range(self.datasetones.shape[0]):
                print(j)
                output = self.__forwardPass(j)
                self.__backwardPass(output, j)

    def predict(self, datas):
        # print(datas)
        datas = self.__builtMatrik(datas)
        # print(datas)
        predicts = np.zeros(datas.shape[0])
        for i in range(datas.shape[0]):
            inp = datas[i]
            for j in range(len(self.weights)-1):
                # print()
                # print(inp.shape)
                # print(self.weights[j].shape)
                net = inp.dot(self.weights[j])
                activation = 1 / (1 + np.exp(-net)) # sigmoid
                ones = np.ones(activation.shape[0]+1) # add matrix one
                ones[1:] = activation
                inp = ones
            net = inp.dot(self.weights[j+1])
            activation = 1 / (1 + np.exp(-net)) # sigmoid
            predicts[i] = np.argmax(activation)

        return predicts

    def to_numeric(self, categorical):
        num = np.zeros(categorical.shape[0])
        for j in range(categorical.shape[0]):
            num[j] = np.argmax(categorical[j])
        return num

    def score(self, target, predict):
        target = self.to_numeric(target)
        # predict = self.to_numeric(predict)
        truth = target == predict
        t = 0
        f = 0
        for i in truth:
            if i:
                t+=1
            else:
                f+=1
        return t/truth.shape[0]


data = load_digits()
categorical = to_categorical(data.target)

img = plt.imread('img/img3.png')[:,:,0]
flatimg = img.flatten()
flatimg = flatimg.reshape((1,flatimg.shape[0]))


X_train, X_test, y_train, y_test = train_test_split(data.data, categorical, test_size=0.3, random_state=42)

a = ArtificialNeuralNetwork(X_train, y_train, 10, lr=0.05, hidden_layer=[64,64])

a.train()

predicts = a.predict(X_test)

# predicts = a.predict(flatimg) # menggunakan input citra

print(predicts)

print('score = ', a.score(y_test, predicts))
