from sklearn.cluster import KMeans
import numpy as np
import loadCSV
from sklearn.model_selection import train_test_split

datasets = loadCSV.LoadCSV('Mall_Customers.csv', ',')
data = datasets[:,:]

data_training, data_test = train_test_split(data, test_size=0.30, random_state=0)#Membagi data menjadi data training dan data testing
est = KMeans(2)
est.fit(data)#Membuat model KMeans
y_kmeans = est.predict(data)#Memprediksi model berdasarkan data
centroid = est.cluster_centers_#Memperoleh nilai centroid
label = est.labels_#Memperoleh class/label
c0 = []
c1 = []
print("Centroid = ",centroid)
print("Label = ",label)

for i in range(len(data)):#Looping untuk memisahkan data berdasarkan cluster 1 dan 0
    if (label[i] == 0):
        c0.append(data[i])
    else:
        c1.append(data[i])
C0 = np.array(c0)
C1 = np.array(c1)
a=0
b=0
s=0
for i in range(len(C0[1:])): #Menghitung jarak antara data dengan data pada cluster terdekat (rumus ecludian distance)
    a+=(C0[0]-C0[1:][i])**2

for i in range(len(C1)):#Mengitung jarak antara data dengan data pada cluster terjauh
    b+=(C0[0]-C1[i])**2

a=a**0.5
a=np.average(a)

b=b**0.5
b=np.average(b)

if a < b:#rumus silhoutte coefficient
    s = 1-(a/b)
elif a > b:
    s = (b/a)-1
else:
    s=0
print("Accuracy pada a[0] = ",s)


