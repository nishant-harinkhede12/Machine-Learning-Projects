import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib
import warnings
warnings.filterwarnings("ignore") # Don't want to see the warnings in the notebook

# from sklearn import svm

data = pd.read_csv(r'E:\Machine Learning\Project_ TASK\TASK-22_ 4 Imp Sub Tasks\avocado.csv')

data.isnull().sum()

data = data.drop(['Unnamed: 0'], axis = 1)
data = data.rename(index=str, columns={"4046" : "Small Hass", "4225" : "Large Hass","4770" : "XLarge Hass" })
data['Date'] = pd.to_datetime(data['Date'])

avocodo = data.iloc[:,[1,3,4,5,6,7,8,9,10,11,12]]

avocodo.info()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

A = avocodo.loc[:,'Small Hass':'XLarge Bags'] = sc.fit_transform(avocodo.loc[:,'Small Hass':'XLarge Bags'])
B = pd.get_dummies(avocodo[['type','year','region']],drop_first=True)

x = np.concatenate([A,B], axis = 1)

y = avocodo.iloc[:,0].values
y = sc.fit_transform(y.reshape(-1, 1))
#---------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#---------------------------------------------------------------------------------------------------------------------------

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
#import the LinearRegression class from sklearn package
regressor = LinearRegression()
#creaet the regressor object for LineareRegression
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
#---------------------------------------------------------------------------------------------------------------------------

# we can confirm the R2 value (moreover, get the R2 Adj.value) of the model by statsmodels library of python
import statsmodels.api as sm
x_train = sm.add_constant(x_train) # adding a constant
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

# #OR
# import statsmodels.api as sm
# regressor_OLS = sm.OLS(endog=y_train, exog=x_train).fit()

# regressor_OLS.summary()

-----------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 2000, random_state = 0)
regressor.fit(x, y)