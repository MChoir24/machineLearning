import numpy as np
import csv
from sklearn.model_selection import train_test_split
import math
import random

def openCsv(file, delimiterSymbol):
    data = []
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = delimiterSymbol, quotechar='|')
        for row in spamreader:
            data.append(row)
    return np.array(data[1:], dtype=float)

def centroid(dataset, numOfClass):
    fitur = []
    centroid = []
    for i in range(dataset.shape[1]):
        fitur.append(dataset[:,i])

    for i in range(numOfClass):
        temp = []
        for j in range(len(fitur)):
            temp.append(random.choice(fitur[j])) #menentukan centroid awal dengan random
        centroid.append(temp)

    return centroid

def euclidean(data, centroid):
    result = []
    distance = []
    for i in range(len(centroid)):
        result.append((data[i] - centroid[i]) ** 2)
    return sum(result) ** 0.5 #nilai kembalian dari perhitungan jarak

def kmeans(data, centroid, epoch):
    for g in range(epoch): #perulangan dengan epoch yang ditentukan
        cluster = [ [] for i in range(len(centroid)) ] #list untuk menampung hasil cluster
        for i in range(len(dataset)):
            distance = [] #list menampung setiap jarak
            for j in range(len(centroid)):
                distance.append(euclidean(dataset[i], centroid[j]))
            min = np.argmin(distance) #mencari nilai terkecil antara centroid dan data
            for l in range(len(centroid)):
                if min == l:
                    cluster[l].append(dataset[i]) #cek jika sama dengan nilai cluster akan di tambahkan ke list clusternya

        cluster = np.array(cluster)
        for h in range(len(centroid)):
            centroid[h] = np.average(cluster[h], axis=0) #update centroid setiap epoch
    return centroid, cluster #nilai kembalian berupa centroid dan cluster

def predic(data, centroid):
    cluster = [ [] for i in range(len(centroid)) ]

    for i in range(len(data)):
        distance = []
        for j in range(len(centroid)):
            distance.append(euclidean(dataset[i], centroid[j]))
            min = np.argmin(distance)
            if min == j:
                cluster[j].append(dataset[i])

    return np.array(cluster)

def silhoutteScore(cluster):
    result = 0
    distance = []
    for i in range(len(cluster[0])):
        for j in range(i, len(cluster[0])-1):
            distance.append(
                # euclidean(cluster[0][i], cluster[0][j+1])
                np.linalg.norm(cluster[0][i] - cluster[0][j+1])
            )
    a = np.average(distance)

    anotherDistance = []
    for i in range(len(cluster[0])):
        temp = []
        for j in range(len(cluster[1])):
            temp.append(
                # euclidean(cluster[0][i], cluster[1][j])
                np.linalg.norm(cluster[0][i] - cluster[1][j])
            )
        anotherDistance.append(np.average(temp))
    b = np.min(anotherDistance)

    result = ( b - a) / np.maximum(a, b)
    print(result)
dataset = openCsv('Mall_Customers.csv', ',') #get dataset

centroid = centroid(dataset, 2) # set early centroid
newCenter, cluster = kmeans(dataset, centroid, 10) #training data
predict = predic(dataset, newCenter) #testing data
silhoutteScore(cluster)
#print(predict)


"""
    Penjelasan:

    fungsi openCsv => fungsi yang digunakan untuk mengambil dataset dari file csv
    kemudian dimasukkan ke dalam array dan menjadi nilai kembalian

    fungsi centroid => fungsi yang digunakan untuk menentukan nilai titik tengah setiap cluster

    fungsi euclidean => fungsi yang digunakan untuk menghitung jarak setiap data dengan centroid

    fungsi kmeans => fungsi yang digunakan untuk menentukan setiap data termasuk dalam cluster yang memiliki jarak
    terpendek dengan centroid

    fungsi predict => fungsi yang digunakan untuk memprediksi data akan termasuk cluster yang mana dengan centroid yang baru

    fungsi silhoutteScore => fungsi yang digunakan untuk menghitung akurasi dari ketepatan prediksi
"""
