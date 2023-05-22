import pandas as pd
from sklearn import preprocessing
import numpy as np


X_train = pd.read_csv('train/x_train.csv', index_col=0)
y_train = pd.read_csv('train/y_train.csv', index_col=0)
X_test = pd.read_csv('test/x_test.csv', index_col=0)
y_test = pd.read_csv('test/y_test.csv', index_col=0)

le=preprocessing.LabelEncoder()
one=preprocessing.OneHotEncoder()
X_train.seller_type=le.fit_transform(X_train.seller_type)
X_test.seller_type=le.fit_transform(X_test.seller_type)

X_train.owner=le.fit_transform(X_train.owner)
X_test.owner=le.fit_transform(X_test.owner)

X_train.name=le.fit_transform(X_train.name)
X_test.name=le.fit_transform(X_test.name)

X_train.to_csv('train/x_train.csv')
X_test.to_csv('test/x_test.csv')


print('*Data preprocessed.*')


