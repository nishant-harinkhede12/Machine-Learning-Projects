import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib
# import warnings
# warnings.filterwarnings("ignore") # Don't want to see the warnings in the notebook

# from sklearn import svm

data = pd.read_csv(r'E:\Machine Learning\Project_ TASK\TASK-22_ 4 Imp Sub Tasks\avocado.csv')

data.isnull().sum()

data = data.drop(['Unnamed: 0'], axis = 1)
data = data.rename(index=str, columns={"4046" : "Small Hass", "4225" : "Large Hass","4770" : "XLarge Hass" })
data['Date'] = pd.to_datetime(data['Date'])

dummies = pd.get_dummies(data[['type','year','region']],drop_first=True)

data.info()
x = pd.concat([data.iloc[:,[3,4,5,6,7,8,9]],dummies],axis = 1)
y = data.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train.iloc[:, [0,1,2,3,4,5,6]] = sc_x.fit_transform(x_train.iloc[:, [0,1,2,3,4,5,6]])
x_test.iloc[:, [0,1,2,3,4,5,6]] = sc_x.transform(x_test.iloc[:, [0,1,2,3,4,5,6]])

#importing ML models from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#to save time all models can be applied once using for loop
regressors = {
    'Linear Regression' : LinearRegression(),
    'Decision Tree' : DecisionTreeRegressor(),
    'Random Forest' : RandomForestRegressor(),
    'Support Vector Machines' : SVR(gamma=1),
    'K-nearest Neighbors' : KNeighborsRegressor(n_neighbors=1),
    'XGBoost' : XGBRegressor()
}
results=pd.DataFrame(columns=['MAE','MSE','R2-score'])
for method,func in regressors.items():
    model = func.fit(x_train,y_train)
    pred = model.predict(x_test)
    results.loc[method]= [np.round(mean_absolute_error(y_test,pred),3),
                          np.round(mean_squared_error(y_test,pred),3),
                          np.round(r2_score(y_test,pred),3)
                         ]
    
results.sort_values('R2-score',ascending=False).style.background_gradient(cmap='Greens',subset=['R2-score'])
