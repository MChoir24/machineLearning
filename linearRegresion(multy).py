import numpy as np
from numpy.linalg import inv

# x = [1, 2, 4, 3, 5]
y = [1, 3, 3, 2, 5]

x = np.array([[1, 1], [1, 2], [1, 4], [1, 3], [1, 5]])
y = np.array(y)
xt = np.transpose(x)
print(x.shape)
print(xt.shape)
# xi = inv(x)
temp = inv(xt.dot(x))
w = temp.dot(xt).dot(y)
print(w)
