import numpy as np


def sortData(data, target):
    ind = np.argsort(data)
    data = np.sort(data)
    target = target[ind]
    return data, target

def median(data):
    # data, target = sortData(data, target)
    median = []
    for i in range(len(data)-1):
        median.append((data[i] + data[i+1]) / 2)
    return median

def gini(arg):
    pass

data = np.array([5.1, 4.9, 7, 6.4, 6.3, 5.8])
target = np.array([0, 0, 1, 1, 2, 2])

dataS, targetS = sortData(data, target)
medians = median(dataS)

# print(medians)
