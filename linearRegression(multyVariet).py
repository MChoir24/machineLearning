import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


boston = datasets.load_boston()
boston1 = boston.data[:50, :2]
bostonTarget = boston.target[:50]
# print(boston1[:, 0])
# print(boston1[:, 1])

plt.plot(boston1[:, :1], bostonTarget, 'ro')
# plt.axis([0, 6, 0, 6])
plt.show()
