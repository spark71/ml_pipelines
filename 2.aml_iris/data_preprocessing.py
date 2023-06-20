import pandas as pd
from sklearn.preprocessing import StandardScaler



X_train = pd.read_csv('train/x_train.csv', index_col=0)
y_train = pd.read_csv('train/y_train.csv', index_col=0)
X_test = pd.read_csv('test/x_test.csv', index_col=0)
y_test = pd.read_csv('test/y_test.csv', index_col=0)


std_slc = StandardScaler()
std_slc.fit(X_train)
X_train_std = pd.DataFrame(std_slc.transform(X_train))
X_test_std =  pd.DataFrame(std_slc.transform(X_test))

X_train_std.to_csv('train/x_train.csv')
X_test_std.to_csv('test/x_test.csv')

print('!')

