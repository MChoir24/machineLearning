import numpy as np
import csv


def LoadCSV(fileName, deli):
    data = []
    with open(fileName, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=deli, quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            data.append(row)
    return np.array(data[1:], dtype=float)
