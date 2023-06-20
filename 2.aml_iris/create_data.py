import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split 
from sklearn import metrics 

data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['Species'] = data.target

X=df.iloc[:,0:4]
Y=df["Species"]

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
print("Train Shape",X_train.shape)
print("Test Shape",X_test.shape)
X_train.to_csv('train/x_train.csv')
y_train.to_csv('train/y_train.csv')
X_test.to_csv('test/x_test.csv')
y_test.to_csv('test/y_test.csv')




