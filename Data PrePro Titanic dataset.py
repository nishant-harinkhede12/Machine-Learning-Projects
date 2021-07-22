import pandas as pd
import numpy as np

data = pd.read_csv(r"E:\Machine Learning\Project_ TASK\TASK-13\DATASET\train.csv")

data.isnull().sum()

#Data Cleaning

# del data['Name']
# del data['Ticket']
# del data['Fare']

Titanic = data.iloc[:,[0,1,2,4,5,6,7,9,11]]

Titanic.describe() 
Titanic.isnull().sum()
#----------------------------------------------------------------------------------
x = Titanic.iloc[:,[0,2,3,4,5,6,7,8]].values
y = Titanic.iloc[:,1].values

#----------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
#imputer = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=(786))#, default = 'None' )

imputer  = imputer.fit(x[:,3:9]) 

x[:,3:9] = imputer.transform(x[:,3:9])
#----------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

LE.fit_transform(x[:,2]) 

x[:,2] = LE.fit_transform(x[:,2]) 

#----------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

LE.fit_transform(x[:,7]) 

x[:,7] = LE.fit_transform(x[:,7]) 

#----------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#--------------------------------------------------------------

#feature scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)