import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.nan)

data = pd.read_csv(r'E:\Machine Learning\3.June_21\1st june\TASK 12 -  TASK 17\TASK-16\kc_house_data.csv')
data 

space = data['sqft_living']
price = data['price']

sp_pp = data[['bedrooms','bathrooms','sqft_living','waterfront']]

x = np.array(space).reshape(-1,1)
y = np.array(price)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

plt.scatter(x_train, y_train,color='b')
plt.plot(x_train,reg.predict(x_train),color='r')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

plt.scatter(x_test, y_test, color='r')
plt.plot(x_test,reg.predict(x_test),color='b')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#_________________________________________________________________________________
#Multiplr regression 'Task 17'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
np.set_printoptions(threshold=np.nan)

data_1 = pd.read_csv(r'E:\Phase 1\Machine Learning\Task\Task 16&17\kc_house_data.csv')
data_1

data_1.info()

data_1 = data_1.drop(['id','date'],axis = 1)
data_1.info()
#_________________________________________________________________________________
with sns.plotting_context("notebook",font_scale=1):
    g = sns.pairplot(data_1[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',size=3)
g.set(xticklabels=[]);

x = data_1.iloc[:, 1:]
y = data_1.iloc[:,0]

#z = data_1.iloc[:,:4]
#_________________________________________________________________________________
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train,y_train)

y_prid = reg.predict(x_test)
#_________________________________________________________________________________

x_1 = pd.get_dummies(x)
import statsmodels.formula.api as sm
x_2 = np.append(arr = np.ones((21613,1)).astype(int),values= x_1, axis = 1)

import statsmodels.api as sm
x_opt = x_2[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

regression_ols = sm.OLS(endog=y, exog=x_opt).fit()
regression_ols.summary()


import statsmodels.api as sm
x_opt = x_2[:, [0,1,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

regression_ols = sm.OLS(endog=y, exog=x_opt).fit()
regression_ols.summary()

#OR#__________________________________________________

#Backward Elimination
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((21613,18)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return a
 
SL = 0.05
X_opt = x_2[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]

print(statsmodels.__version__)

X_Modeled = backwardElimination(X_opt, SL)
