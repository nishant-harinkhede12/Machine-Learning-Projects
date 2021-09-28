#Project picked from below link
# https://thecleverprogrammer.com/2021/08/19/water-quality-analysis/

import pandas as pd
import numpy as na
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/amankharwal/Website-data/master/water_potability.csv'

data = pd.read_csv(url)
data.head()

data.isnull().sum()

data1 = data.dropna()

data1.isnull().sum()

# 491+781+162
# 1434 + 2011

sns.countplot(data.Potability)
# pip install plotly

import plotly.express as px
data = data
figure = px.histogram(data, x = "ph", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: PH")
figure.show()

figure = px.histogram(data, x = "Hardness", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Hardness")
figure.show()

##############################################################
#below matrics suggest that there is no linear relationship in dataset
##############################################################
cor = data1.corr()

columns = cor[cor["Potability"]>0.7]["Potability"]
columns_1 = cor[cor["Potability"]<(-0.7)]["Potability"]

C = pd.concat([columns,columns_1],axis=1)
columns_list = C.index
columns_list

data2 = pd.DataFrame(data1,columns = columns_list)

##############################################################

x = data1.iloc[0:,0:9]
y = data1.iloc[0:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
sc.transform(X_test)

##############################################################
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("accuracy score of svm = ",ac)
###########################################################
from sklearn.svm import SVC
classifier = SVC(kernel='sigmoid', random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("accuracy score of svm = ",ac)
###########################################################
from sklearn.svm import SVC
classifier = SVC(kernel='polynomial', random_state=0)
# classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("accuracy score of svm = ",ac)
###########################################################
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_test,y_pred)
print("accuracy score of LogisticRegression = ",ac1)
###########################################################
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
ac2 = accuracy_score(y_test,y_pred)
print("accuracy score of DecisionTreeClassifier = ",ac2)

###########################################################

from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
ac2 = accuracy_score(y_test,y_pred)
print("accuracy score of DecisionTreeClassifier = ",ac2)
###########################################################


# from sklearn.GiniIndex import RandomForestClassifier
# regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
# regressor.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# from sklearn.metrics import accuracy_score
# ac2 = accuracy_score(y_test,y_pred)
# print("accuracy score of DecisionTreeClassifier = ",ac2)
###########################################################

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
ac3 = accuracy_score(y_test,y_pred)
print("accuracy score of naive_bayes = ",ac3)
###########################################################
# pycaret is library which will apply all classification algorithem
###########################################################
# from pycaret.classification import *
# clf = setup(data1, target = "Potability", silent = True, session_id = 786)
# compare_models()

# model = create_model("rf")
# predict = predict_model(model, data=data1)
# predict.head()