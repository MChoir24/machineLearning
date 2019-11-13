import numpy as np
import matplotlib.pyplot as plt


def rata2(x, col=None):
    sumRow = []
    for j in range(len(x[0])):
        sumCol = 0
        for k in range(len(x)):
            sumCol += x[k][j]
        sumCol = sumCol / len(x)
        sumRow.append(sumCol)
    if col is None:
        result = 0
        for i in sumRow:
            result += i
    elif col is not None and type(col) is int:
        result = sumRow[col]

    return result


def varian(x):
    sigma = 0
    for i in range(len(x)):
        sigma += (x[i][0]-rata2(x, 0))
    result = sigma / (len(x) - 1)
    return result


def cov(x):
    sigma = 0
    for i in range(len(x)):
        perkalian = 1
        for j in range(len(x[0])):
            perkalian *= (x[j][i])
        sigma += perkalian
    result = sigma / (len(x) - 1)
    return result


def w1(x):
    return cov(x) / varian(x)


def w0(x, y):
    return rata2(y) - (w1(x, y) * rata2(x))


data = np.array([[1, 1],
                [2, 3],
                [4, 3],
                [3, 2],
                [5, 5]])
x = [1, 2, 4, 3, 5]
y = [1, 3, 3, 2, 5]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 6])
plt.show()
