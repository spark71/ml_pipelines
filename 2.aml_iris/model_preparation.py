# import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


X_train = pd.read_csv('train/x_train.csv', index_col=0)
y_train = pd.read_csv('train/y_train.csv', index_col=0)
X_test = pd.read_csv('test/x_test.csv', index_col=0)

model = LogisticRegression()
model.fit(X_train,y_train)

y_predict=pd.DataFrame(model.predict(X_test))
y_predict.to_csv('pred/y_pred.csv')
# save_file = 'pred/lr_model.sav'
# pickle.dump(model, open(save_file, 'wb'))
