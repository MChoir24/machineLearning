import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


data = load_digits()
dataimg = []
a = 0
for i in data.data:
    dataimg.append(i.reshape((8,8)))
    plt.imsave('img/img%s.png'%a, i.reshape((8,8)), cmap='gray')
    plt.close()
    a+=1
dataimg = np.array(dataimg)
print(dataimg.shape)
