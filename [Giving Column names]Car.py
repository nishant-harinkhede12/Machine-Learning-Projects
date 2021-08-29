import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r'E:\Machine Learning\Project_ TASK\TASK-39 to TASK-49\car-evaluation-data-set_ TASK 41\car_evaluation.csv')

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data.columns = col_names

data.info()
data.isnull().sum()
# data['buying'].value_counts()

for count in col_names:
    print(data[count].value_counts())


x = data.drop(['class'],axis = 1).values
y = data['class'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(sparse=False)

x_train = OHE.fit_transform(x_train)
x_test = OHE.transform(x_test)
###########################################################
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
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
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
ac3 = accuracy_score(y_test,y_pred)
print("accuracy score of naive_bayes = ",ac3)
###########################################################
# Fitting Random Forest Regression to the dataset
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

