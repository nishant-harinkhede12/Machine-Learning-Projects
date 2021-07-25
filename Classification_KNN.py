import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

df = pd.read_csv(r'E:\Phase 1\My_Codes\Machine Learning\Classification KNN\Classified Data',index_col=0)
print(df.head())
print(df.isnull().sum())
###############################################################################
l=list(df.columns)
l[0:len(l)-2]

for i in range(len(l)-1):
    sns.boxplot(x='TARGET CLASS',y=l[i], data=df)
    plt.figure()
    
for i in range(len(l)-1):
    sns.scatterplot(x='TARGET CLASS',y=l[i], data=df)
    plt.figure()

###############################################################################
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()

x = df.drop(['TARGET CLASS'],axis = 1)
y = df['TARGET CLASS']

x=SC.fit_transform(x)
###############################################################################
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)
###############################################################################
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()

KNN.fit(x_train,y_train)

y_pred = KNN.predict(x_test)
###############################################################################
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_test,y_pred)

print(cm)
print(ac)
