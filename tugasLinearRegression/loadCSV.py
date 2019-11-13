import numpy as np
import csv


def loadCSV(fileName):
    data = []
    with open(fileName, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            data.append(row)
    return np.array(data[1:], dtype=float)
