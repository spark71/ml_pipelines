import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


X_train = pd.read_csv('train/x_train.csv', index_col=0)
y_train = pd.read_csv('train/y_train.csv', index_col=0)
X_test = pd.read_csv('test/x_test.csv', index_col=0)

LR=LinearRegression()
LR.fit(X_train, y_train)

y_predict=pd.DataFrame(LR.predict(X_test))
y_predict.to_csv('pred/y_pred.csv')
save_file = 'pred/lr_model.sav'
pickle.dump(LR, open(save_file, 'wb'))

print('*Model has been prepared in pred/.*')