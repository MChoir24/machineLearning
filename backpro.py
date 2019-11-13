import numpy as np

data = np.array([5.1, 3.5, 1.4, 0.2])
w = np.array([[0.1,0.3,0.4],[]])

def forward(data, w):
    net = data.dot(w)
    net = sum(net)
    aktivasi = 1/(1+(np.exp**net))
    return aktivasi

def backward(data, w, ok, tk, yj):
    fok = ok * (1 - ok)
    sok = (tk - ok) * fok
    dV = e * sok * yj
    V = V + d
