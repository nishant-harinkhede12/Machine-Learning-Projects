import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

url ='https://raw.githubusercontent.com/nishant-harinkhede12/Machine-Learning-Projects/main/wine.data.csv'

df = pd.read_csv(url)

#Data Visualization
for c in df.columns[1:]:
    df.boxplot(c,by = 'Class')
    plt.title("{}\n".format(c))

df.info()
plt.scatter(x=df['OD280/OD315 of diluted wines'],y=df['Flavanoids'],c=df['Class'])
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation",fontsize=15)
plt.xlabel("OD280/OD315 of diluted wines",fontsize=15)
plt.ylabel("Flavanoids",fontsize=15)

#Correlation Matrix

Source = df
source_corr = Source.corr()
ax = sns.heatmap(source_corr,
            xticklabels=source_corr.columns,
            yticklabels=source_corr.columns,
            annot = True,
            cmap ="RdYlGn")

x=df.drop('Class',axis = 1)
y=df.Class

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=42)

from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()

GNB.fit(x_train,y_train)
y_pred = GNB.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

CR = classification_report(y_pred,y_test)
print(CR)

CM = confusion_matrix(y_pred,y_test)
print(CM)

AC = accuracy_score(y_test,y_pred)
print(AC)
