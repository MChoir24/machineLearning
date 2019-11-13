from sklearn import datasets
import pandas as pd

data = datasets.load_boston()
print(data)

df = pd.DataFrame(data=data['data'], columns = data['feature_names'])
df.to_csv('boston.txt', sep = ',', index = False)
