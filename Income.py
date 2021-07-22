import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

data_raw = pd.read_csv(r'E:\Machine Learning\Project_ TASK\TASK-37\adult-dataset\adult.csv')

data = pd.read_csv(r'E:\Machine Learning\Project_ TASK\TASK-37\adult-dataset\adult.csv', header=None, sep=',\s')

data.info()

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

data.columns = col_names # this is very important step were we changed the name of columns

data.info()
#---------------------------------------------------------------------------------------------------------------------------
data.isnull().sum()

data['workclass'].value_counts()
data['occupation'].value_counts()
data['native_country'].value_counts()

data= data.iloc[:, 0:16].replace('?',np.NaN) 

data.isnull().sum()

data['workclass'].fillna(data['workclass'].mode()[0],inplace=True)
data['occupation'].fillna(data['occupation'].mode()[0],inplace=True)
data['native_country'].fillna(data['native_country'].mode()[0],inplace=True)

data['income'].value_counts()
data= data.iloc[:, 0:15].replace('<=50K', '0')
data= data.iloc[:, 0:15].replace('>50K','1') 
#---------------------------------------------------------------------------------------------------------------------------
x = data.drop(['income'],axis = 1).values
y = data['income'].values

#---------------------------------------------------------------------------------------------------------------------------
categorical = [a for a in data.columns if data[a].dtype=='O']
numerical = [a for a in data.columns if data[a].dtype!='O']

y1 = data[categorical].info()

# from sklearn.preprocessing import OneHotEncoder
# OHE = OneHotEncoder()

# OHE.fit_transform(x[:,[1,3,5,6,7,8,9,13]]) #[1,3,5,6,7,8,9,13]
# x[:,[1,3,5,6,7,8,9,13]] = OHE.fit_transform(x[:,[1,3,5,6,7,8,9,13]]) 
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

LE.fit_transform(x[:,1]) #[1,3,5,6,7,8,9,13]
x[:,1] = LE.fit_transform(x[:,1]) 

LE.fit_transform(x[:,3]) #[1,3,5,6,7,8,9,13]
x[:,3] = LE.fit_transform(x[:,3]) 

LE.fit_transform(x[:,5]) #[1,3,5,6,7,8,9,13]
x[:,5] = LE.fit_transform(x[:,5]) 

LE.fit_transform(x[:,6]) #[1,3,5,6,7,8,9,13]
x[:,6] = LE.fit_transform(x[:,6]) 

LE.fit_transform(x[:,7]) #[1,3,5,6,7,8,9,13]
x[:,7] = LE.fit_transform(x[:,7]) 

LE.fit_transform(x[:,8]) #[1,3,5,6,7,8,9,13]
x[:,8] = LE.fit_transform(x[:,8]) 

LE.fit_transform(x[:,9]) #[1,3,5,6,7,8,9,13]
x[:,9] = LE.fit_transform(x[:,9]) 

LE.fit_transform(x[:,13]) #[1,3,5,6,7,8,9,13]
x[:,13] = LE.fit_transform(x[:,13]) 

#---------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#--------------------------------------------------------------

#feature scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
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

