
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sdv.single_table import GaussianCopulaSynthesizer
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sdv.metadata import SingleTableMetadata


data = pd.read_csv('BIKE DETAILS.csv')

data.ex_showroom_price.fillna(np.round(data.ex_showroom_price.mean(),2),inplace=True)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)

synthesizer = GaussianCopulaSynthesizer(metadata,  numerical_distributions={'name':'norm', 'selling_price':'norm', 'year':'norm',
                                                                            'seller_type':'norm', 'owner':'uniform', 'km_driven':'norm',
                                                                            'ex_showroom_price':'norm'})

synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=10000)
x=synthetic_data.drop('selling_price',axis=1)
y=synthetic_data.selling_price

X_train ,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
X_train.to_csv('train/x_train.csv')
y_train.to_csv('train/y_train.csv')
X_test.to_csv('test/x_test.csv')
y_test.to_csv('test/y_test.csv')

print('*Data created in train and test.*')





