import numpy as np
import matplotlib.pyplot as plt


def ecludian(x, centeroid):
    tempx = (x[0] - centeroid[0]) ** 2
    tempy = (x[1] - centeroid[1]) ** 2
    result = (tempx + tempy) ** 0.5
    # print(x[0], centeroid[0])
    # print(x[1], centeroid[1])
    return result


def kmean(data, center, n):
    label = np.zeros(len(data))
    print(center)
    for epho in range(n):
        clus0 = []
        clus1 = []
        for i in range(len(data)):
            temp0 = ecludian(data[i], center[0])
            temp1 = ecludian(data[i], center[1])
            label[i] = np.argmin([temp0, temp1])
            # print(i, temp0, temp1)
            if label[i] == 1:
                clus1.append(data[i])
            else:
                clus0.append(data[i])

        clus0 = np.array(clus0)
        clus1 = np.array(clus1)
        print(clus0)
        center[0] = np.average(np.array(clus0), axis=0)
        center[1] = np.average(np.array(clus1), axis=0)
        plt.plot(clus0[:, 0], clus0[:, 1], 'ro', clus1[:, 0],
                 clus1[:, 1], 'bs', center[:, 0], center[:, 1], 'g^')
        plt.savefig('epho%s.png' % epho)
        plt.close()
        # plt.show()
        print(center)
    return center, clus0, clus1, label


def predict(x, center):
    label = np.zeros(len(x))
    for i in range(len(x)):
        temp0 = ecludian(x[i], center[0])
        temp1 = ecludian(x[i], center[1])
        label[i] = np.argmin([temp0, temp1])
    return label


data = np.array([[7, 5], [5, 7], [7, 7], [3, 3], [4, 6],
                 [1, 4], [0, 0], [2, 2], [8, 7], [6, 8]])
center = np.array([[4, 6], [5, 5]], dtype=float)

# print(ecludian(data[0], center[0]))
# print(ecludian(data[0], center[1]))
newCenter, clus0, clus1, label = kmean(data, center, 1)
newLabel = predict(data, newCenter)
print(label)
