import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/akjadon/HH/master/Python/DS_tutorials_all/Data/house_rental_data.csv.txt'
data = pd.read_csv(url, sep = ',',index_col='Unnamed: 0')
# data = pd.read_csv('https://raw.githubusercontent.com/akjadon/HH/master/Python/DS_tutorials_all/Data/house_rental_data.csv.txt',index_col='Unnamed: 0'))

data.head()
data.isnull().sum()
#--------------------------------------------------------------------------------------------
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors()
nn.fit(data)
nn.kneighbors(data[:1])

a = data.iloc[[ 0, 354, 319, 379, 269]]
#--------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
data_scale = ss.fit_transform(data.values)
nn.kneighbors(data_scale[:1])

b = data.iloc[[609, 404, 620, 371, 440]]
#--------------------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
data_mm = mm.fit_transform(data.values)

mm.fit(data_mm)
nn.kneighbors(data_mm[:1])

c = data.iloc[[609, 404, 620, 371, 440]]
