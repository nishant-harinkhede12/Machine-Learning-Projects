import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()

boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)

boston_df.info()

boston_df['MEDV'] = boston.target

boston_df.head()

x = boston_df[['RM']]
y = boston_df['MEDV']

from sklearn.linear_model import  LinearRegression
LR = LinearRegression()

LR.fit(x,y) 
 
print('Cofficient:', LR.coef_)
print('Intercept:',LR.intercept_)

y_pred = LR.predict(x)

print("Predicted vALUE:", y_pred[:4])

print("Actusl Values:\n",boston_df[['RM','MEDV']][:4])

X = boston_df.drop(['MEDV'],axis = 1)

LR.fit(X,y) 

print('Cofficient:', LR.coef_)
print('Intercept:',LR.intercept_)

Y_pred = LR.predict(X)

print("Predicted vALUE:", Y_pred[:4])

print("Actusl Values:\n",boston_df[:4])


import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(x,y,c='b')
plt.plot(x,y_pred,c='r',linewidth=3)
plt.show()


from sklearn.metrics import mean_squared_error, r2_score

print("Mean squared error: %.2f" % mean_squared_error(y, y_pred))

print('Variance score: %.2f' % r2_score(y, y_pred))

print("Mean squared error: %.2f" % mean_squared_error(y, Y_pred))

print('Variance score: %.2f' % r2_score(y, Y_pred))