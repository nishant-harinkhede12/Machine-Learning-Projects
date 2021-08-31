import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'E:\Machine Learning\14th june\Social_Network_Ads.csv')

data.info()

data.isnull().sum()


x = data.iloc[:,[1,2,3]].values
#x = data.iloc[: ,1:4].values
y = data.iloc[: ,-1].values

#x.info()
#########################################################
from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

labelencoder_x.fit_transform(x[:,0]) 

x[:,0] = labelencoder_x.fit_transform(x[ : ,0]) 
###########################################################################
from sklearn.preprocessing import StandardScaler

reg = StandardScaler()

x = reg.fit_transform(x)
#x_test = reg.transform(x_test)
###########################################################################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3, random_state = 45)

##############################################################
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("accuracy score of svm = ",ac)
###########################################################
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_test,y_pred)
print("accuracy score of LogisticRegression = ",ac1)
###########################################################
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
ac2 = accuracy_score(y_test,y_pred)
print("accuracy score of DecisionTreeClassifier = ",ac2)
###########################################################
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
ac3 = accuracy_score(y_test,y_pred)
print("accuracy score of naive_bayes = ",ac3)
###########################################################
