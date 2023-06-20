import pandas as pd
from sklearn import metrics


prediction=pd.read_csv('pred/y_pred.csv', index_col=0)
y_test = pd.read_csv('test/y_test.csv', index_col=0)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,y_test))
# 