import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

url = 'https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt'

house_data = pd.read_csv(url, index_col='Unnamed: 0')

house_data.info() 

house_data.rename(columns={'Living.Room':'Livingroom'},inplace = True)

x = house_data.iloc[:, :6]
y = house_data.iloc[:, 6]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

import statsmodels.api as sm
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

import statsmodels.api as sm
x_1 = sm.add_constant(x)

regression_ols = sm.OLS(endog=y, exog=x_1).fit()
regression_ols.summary()

#OR
# x_1 = pd.get_dummies(x)
# import statsmodels.formula.api as sm
# x_2 = np.append(arr = np.ones((645,1)).astype(int),values= x_1, axis = 1)

# x_opt = x_2[:, 0:]
# import statsmodels.api as sm
# regression_ols = sm.OLS(endog=y, exog=x_opt).fit()
# regression_ols.summary()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MSE = mean_squared_error(y_test,y_pred)
print(MSE)

rsquare = r2_score(y_test,y_pred)
print(rsquare)




# variables = list(house_data.columns)
# y = 'Price'
# x = [var for var in variables if var not in y ]

# import statsmodels.api as sm

# model_simple = sm.OLS(house_data[y],house_data[x]).fit()
# model = sm.OLS(house_data[y],sm.add_constant(house_data[x])).fit()

# print(model_simple.summary())
# print(model.summary())

# drop_var = ['TotalFloor','Bedroom','Livingroom','Bathroom','Price']
# x_new = [var for var in variables if var not in drop_var ]

# model_new = sm.OLS(house_data[y], sm.add_constant(house_data[x_new])).fit()
# print(model_new.summary())

# from sklearn.metrics import mean_squared_error, adjusted_rand_score, mean_absolute_error

# MSE = mean_squared_error()
# MSE.fit(house_data[x],house_data[y])
# print(MSE)
# # AR2 = adjusted_rand_score()
# # AR2.fit(house_data[y],house_data[x])
# #################################################################################


# # function to compute adjusted R-squared
# def adj_r2_score(predictors, targets, predictions):
#     r2 = r2_score(targets, predictions)
#     n = predictors.shape[0]
#     k = predictors.shape[1]
#     return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


# # function to compute different metrics to check performance of a regression model
# def model_performance_regression(model, house_data[x], house_data[y]):
#     """
#     Function to compute different metrics to check regression model performance

#     model: regressor
#     predictors: independent variables
#     target: dependent variable
#     """

#     # predicting using the independent variables
#     pred = model.predict(house_data[x])

#     r2 = r2_score(target, pred)  # to compute R-squared
#     adjr2 = adj_r2_score(house_data[x], target, pred)  # to compute adjusted R-squared
#     rmse = np.sqrt(mean_squared_error(target, pred))  # to compute RMSE
#     mae = mean_absolute_error(target, pred)  # to compute MAE

#     # creating a dataframe of metrics
#     df_perf = pd.DataFrame(
#         {
#             "RMSE": rmse,
#             "MAE": mae,
#             "R-squared": r2,
#             "Adj. R-squared": adjr2,
#         },
#         index=[0],
#     )

#     return df_perf
    
    
#     # Checking model performance on train set
# print("Training Performance\n")
# lin_reg_model_train_perf = model_performance_regression(model, x_train, y_train)
# lin_reg_model_train_perf