from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np

dataTraining = datasets.load_iris()
temp1 = dataTraining.data[0:2, :]
temp2 = dataTraining.data[50:52, :]
temp3 = dataTraining.data[100:102, :]
