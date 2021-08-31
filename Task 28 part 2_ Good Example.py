import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.mlab as mlab

data = pd.read_csv(r'E:\Machine Learning\Project_ TASK\TASK-28\PART-2\heart-disease-prediction-using-logistic-regression\framingham.csv')
data.info()

data = data.drop(['education'], axis = 1)

data.isnull().sum()

#data.dropna(axis=0,inplace=True)

data['cigsPerDay'].fillna(data['cigsPerDay'].mean(),inplace=True)
data['BPMeds'].fillna(data['BPMeds'].mean(),inplace=True)
data['totChol'].fillna(data['totChol'].mean(),inplace=True)
data['BMI'].fillna(data['BMI'].mean(),inplace=True)
data['heartRate'].fillna(data['heartRate'].mean(),inplace=True)
data['glucose'].fillna(data['glucose'].mean(),inplace=True)

from statsmodels.tools import add_constant
data_constant = add_constant(data)
data_constant.head()

# chi-square test

st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=data_constant.columns[:-1]
model=sm.Logit(data.TenYearCHD,data_constant[cols])
result=model.fit()
result.summary()

**#Feature Selection: Backward elemination (P-value approach)

def back_feature_elem (data_frame,dep_var,col_list):
    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(data_constant,data.TenYearCHD,cols)

#Interpreting the results: Odds Ratio, Confidence Intervals and Pvalues

params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))

#Splitting data to train and test split

import sklearn 

data.info()
new_data = data.iloc[:,[1,0,3,8,9,13,14]]
x = new_data.iloc[:,:-1].values
y = new_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3, random_state=45)

from sklearn.linear_model import LogisticRegression
reg =  LogisticRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)


from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
print(confusion_matrix)

accuracy_score = metrics.accuracy_score(y_test,y_pred)
print(accuracy_score)
