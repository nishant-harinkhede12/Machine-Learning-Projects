import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/akjadon/HH/master/Python/DS_tutorials_all/Data/churn.csv.txt'

data = pd.read_csv(url,sep=',', parse_dates=['last_trip_date','signup_date'])
data.info()
data['last_trip_date'].max()
#-------------------------------------------------------------------------------------------
import datetime
datetime.timedelta(30,0,0)
cutoff = data['last_trip_date'].max() - datetime.timedelta(30,0,0)
data['churn'] = (data['last_trip_date'] < cutoff).astype(int)
data.head()

data.isnull().sum()

data['avg_rating_by_driver'].fillna(data['avg_rating_by_driver'].mode()[0],inplace=True)
data['avg_rating_of_driver'].fillna(data['avg_rating_of_driver'].mode()[0],inplace=True)
data['phone'].fillna(data['phone'].mode()[0],inplace=True)

x= data.drop(['churn','last_trip_date','signup_date'],axis = 1).values
y = data['churn'].values

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

x[:,4] = LE.fit_transform(x[:,4])
x[:,5] = LE.fit_transform(x[:,5])
x[:,8] = LE.fit_transform(x[:,8])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)

#---------------------------------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, random_state = 0)
classifier.fit(x_train, y_train)

# Predicting a new result
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac3 = accuracy_score(y_test,y_pred)
print("accuracy score of Random forest regressor = ",ac3)

#---------------------------------------------------------------------------------------------------------------------------
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

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac4 = accuracy_score(y_test,y_pred)
print("accuracy score of naive_bayes = ",ac4)
###########################################################
