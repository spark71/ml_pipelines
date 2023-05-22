
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score


y_test = pd.read_csv('test/y_test.csv', index_col=0)
y_pred = pd.read_csv('pred/y_pred.csv', index_col=0)
# lr_model = pickle.load(open('pred/lr_model.sav', 'rb'))

r=r2_score(y_test,y_pred) * 100
rmse = mean_squared_error(y_pred,y_test)**0.5
print(f'R2: {r}\nRMSE: {rmse}')
print('*Model has been tested.*')

