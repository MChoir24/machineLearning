def ann(data, target, w, tresh):
    w1 = np.zeros(len(w))
    while w != w1:
        for i in range(len(data)):
            temp = w * data[i]
            tempPredict = sum(temp)
            if tempPredict >= tresh:
                tempPredict = 1
            else:
                tempPredict = 0

            if tempPredict > target[i]:
                w = w - data[i]
            elif tempPredict < target[i]:
                w = w + data[i]
            else:
                w = w
    return w

data = np.array([[1,1,1,1],[1,1,0,0],[1,0,1,0]])
target = np.array([0,1,0,0])
w =
