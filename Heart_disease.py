import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
%matplotlib inline
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r'E:\Machine Learning\Project_ TASK\TASK-39 to TASK-49\heart-disease-uci_ TASK 42\heart.csv')

col_names = data.columns

for C in col_names:
    print(data[C].value_counts())
###############################################################################
ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=data)
plt.show()


ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=data, palette="Set3")
plt.show()

ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=data, facecolor=(0, 0, 0, 0), linewidth=5, edgecolor=sns.color_palette("dark", 3))
plt.show()

ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", hue="fbs", data=data, palette="Set3")
plt.show()

ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", hue="exang", data=data, palette="Set3")
plt.show()

data.groupby('sex')['target'].value_counts()
###############################################################################
ax = plt.subplots(figsize=(8,6))
ax = sns.countplot(x="target", data=data, hue = 'sex')
plt.show()

ax = sns.catplot(x="target", col="sex", data=data, kind="count", height=5, aspect=1)
###############################################################################
ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(y="target", hue="sex", data=data)
plt.show()

ax = sns.catplot(y="target", col="sex", data=data, kind="count", height=5, aspect=1)
###############################################################################
cOrElAtIoN = data.corr()
cOrElAtIoN['target'].sort_values(ascending=False)

# target      1.000000
# cp          0.433798
# thalach     0.421741
# slope       0.345877
# restecg     0.137230

plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Heart Disease Dataset')
a = sns.heatmap(cOrElAtIoN, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()

###############################################################################
ax = plt.subplots(figsize=(4, 3))
ax = sns.countplot(x="cp", data=data)
plt.show()

ax = plt.subplots(figsize=(4, 3))
ax = sns.countplot(x="cp", data=data, hue = 'target')
plt.show()

ax = plt.subplots(figsize=(4, 3))
ax = sns.catplot(x="target", col="cp", data=data, kind="count", height=4, aspect=1)
plt.show()
###############################################################################
 ax = plt.subplots(figsize=(5,3))
x = data['thalach']
ax = sns.distplot(x, bins=10)
plt.show()

ax = plt.subplots(figsize=(5,3))
x = data['thalach']
x = pd.Series(x, name="thalach variable")
ax = sns.distplot(x, bins=10)
plt.show()

ax = plt.subplots(figsize=(5,3))
x = data['thalach']
x = pd.Series(x, name="thalach variable")
ax = sns.distplot(x, bins=10, vertical = True)
plt.show()

ax = plt.subplots(figsize=(10,6))
x = data['thalach']
x = pd.Series(x, name="thalach variable")
ax = sns.kdeplot(x, shade = True, color = 'r')
plt.show()

ax = plt.subplots(figsize=(5,3))
x = data['thalach']
ax = sns.distplot(x, rug=True, bins=10)#kde=False,
plt.show()
###############################################################################
ax = plt.subplots(figsize=(4, 3))
sns.boxplot(x="target", y="thalach", data=data)
plt.show()
###############################################################################
num_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target' ]
sns.pairplot(data[num_var], kind='scatter', diag_kind='hist')
plt.show()
###############################################################################
ax = plt.subplots(figsize=(4, 3))
ax = sns.scatterplot(x="age", y="trestbps", data=data)
plt.show()

ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="age", y="trestbps", data=data)
plt.show()
###############################################################################

ax = plt.subplots(figsize=(4, 3))
sns.boxplot(x=data["oldpeak"])
plt.show()