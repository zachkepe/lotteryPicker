import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("megamillions.csv")

print(df.shape)
print(list(df.columns))

print(df.head())

window_length = 7

df1 = df.copy()
df.drop(['Date'], axis=1, inplace=True)

number_of_features = df.shape[1] 

train = df.copy()

train_rows = train.values.shape[0]
train_samples = np.empty([ train_rows - window_length, window_length, number_of_features], dtype=float)
train_labels = np.empty([ train_rows - window_length, number_of_features], dtype=float)
for i in range(0, train_rows-window_length):
    train_samples[i] = train.iloc[i : i+window_length, 0 : number_of_features]
    train_labels[i] = train.iloc[i+window_length : i+window_length+1, 0 : number_of_features]

print(train_samples[0])

