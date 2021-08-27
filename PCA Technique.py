import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

url = 'https://raw.githubusercontent.com/akjadon/HH/master/Python/Clustering/wine.data.csv'

data = pd.read_csv(url)

for c in data.columns[1:]:
    data.boxplot(c,by='Class')#,figsize=(7,4),fontsize=14)
    plt.title("{}\n".format(c))#,fontsize=16)
    plt.xlabel("Wine Class")#, fontsize=16)


plt.scatter(data['OD280/OD315 of diluted wines'],data['Flavanoids'],c=data['Class'])


def correlation_matrix(data):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(data.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Wine data set features correlation\n',fontsize=15)
    labels=data.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(data)

# cor = data.corr()

# data.info()
# columns = cor[cor["Class"]>0.7]["Class"]
# columns_1 = cor[cor["Class"]<(-0.65)]["Class"]

# C = pd.concat([columns,columns_1],axis=1)
# columns_list = C.index
# columns_list

X = data.drop('Class',axis=1)
y = data['Class']

from sklearn.preprocessing import StandardScaler
SC = StandardScaler()

X = SC.fit_transform(X) 

dfx = pd.DataFrame(data=X,columns=data.columns[1:])

from sklearn.decomposition import PCA
pca = PCA(n_components=None)

dfx_pca = pca.fit(dfx)


plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],
            y=dfx_pca.explained_variance_ratio_,
           s=200, alpha=0.75,c='orange',edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
plt.xlabel("Principal components",fontsize=15)
plt.xticks([i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Explained variance ratio",fontsize=15)
plt.show()


dfx_trans = pca.transform(dfx)
dfx_trans = pd.DataFrame(data=dfx_trans)


plt.figure(figsize=(10,6))
plt.scatter(dfx_trans[0],dfx_trans[1],c=data['Class'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Class separation using first two principal components\n",fontsize=20)
plt.xlabel("Principal component-1",fontsize=15)
plt.ylabel("Principal component-2",fontsize=15)
plt.show()

import seaborn as sns
sns.pairplot(dfx_trans)
# from sklearn.cluster import KMeans

# wcss = []
# for i in range(1, 16):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++')
#     kmeans.fit(data)
#     wcss.append(kmeans.inertia_)

# with plt.style.context(('fivethirtyeight')):
#     plt.figure(figsize=(10,6))
#     plt.plot(range(1, 16), wcss)
#     plt.title('The Elbow Method with k-means++\n',fontsize=25)
#     plt.xlabel('Number of clusters')
#     plt.xticks(fontsize=20)
#     plt.ylabel('WCSS (within-cluster sums of squares)')
#     plt.vlines(x=5,ymin=0,ymax=250000,linestyles='--')
#     plt.text(x=5.5,y=110000,s='5 clusters seem optimal choice \nfrom the elbow position',
#              fontsize=25,fontdict={'family':'Times New Roman'})
#     plt.show()