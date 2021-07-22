import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# import seaborn
# %matplotlib inline

data = pd.read_csv(r'E:\Machine Learning\Project_ TASK\TASK-33\creditcard.csv')

data.info()

print(len(data))
d_train_all = data[0:200000]

d_train_1 =  d_train_all[d_train_all['Class']==1]

d_train_0 = d_train_all[d_train_all['Class']==0]

print(len(d_train_1))
print(len(d_train_0))

d_sample = d_train_0.sample(385)
d_train = d_train_1.append(d_sample)
d_train = d_train.sample(frac = 1) 

x_train = d_train.drop(['Time','Class'],axis = 1)
y_train = d_train['Class']

d_test_all = data[200000:]

d_test_1 =  d_test_all[d_test_all['Class']==1]
d_test_0 = d_test_all[d_test_all['Class']==0]

print(len(d_test_1))
print(len(d_test_0))

x_test = d_test_all.drop(['Time', 'Class'],axis=1)
y_test = d_test_all['Class']

from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', random_state=0)
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("accuracy score of svm = ",ac)


# d_fraud = data[data['Class']==1]

# plt.Figure(figsize=(25,20))

# plt.scatter(d_fraud['Time'],d_fraud['Amount'])
# plt.xlim([0,175000])
# plt.ylim([0,2500])
# plt.show()

# d_corr = data.corr()
# rank = d_corr['Class']

# d_rank = pd.DataFrame(rank)
# df_rank = np.abs(d_rank).sort_values(by='Class',ascending=False) 
# df_rank.dropna(inplace=True)
