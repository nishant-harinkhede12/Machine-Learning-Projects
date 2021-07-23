import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/nishant-harinkhede12/EDA-Projects/main/loan_data.csv'
data = pd.read_csv(url)
data_final = pd.get_dummies(data.drop(['purpose'],axis = 1))

# x = data_final.iloc[:, 0:13].values
x = data_final.drop(['not.fully.paid'],axis=1).values
y = data_final.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =1/3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini',max_depth=None)

DT.fit(x_train,y_train)

y_pred=DT.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

cr = classification_report(y_test,y_pred)
print("'classification_report'",cr)

cm = confusion_matrix(y_test,y_pred)
print("'confusion_matrix'",cm)

ac = accuracy_score(y_test,y_pred)
print("accuracy_score",ac)