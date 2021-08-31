import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import copy

# set pyplot parameters to make things pretty
plt.rc('axes', linewidth = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 3, width = 1.5)
plt.rc('ytick.major', size = 3, width = 1.5)

lineardata = pd.read_csv(r'E:Phase 1\Machine Learning\3.June_21\4th june\task\TASK-20_ SGD\linear.csv')

plt.plot('x','y',data=lineardata, marker = '^', linestyle = ' ', label = 'data')
plt.xlabel('x',fontsize = 18)
plt.ylabel('y', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()

x = lineardata['x'].tolist()
y = lineardata['y'].tolist()

# Don't forget - adding ones to the x matrix
xb = np.c_[np.ones((len(x),1)),x]
# calculate linear regression parameters theta using the normal equation
thetaHat = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)

print(thetaHat)

xFit = np.linspace(0,100,num = 200)
xFitb = np.c_[np.ones((len(xFit),1)), xFit]
yFit = xFitb.dot(thetaHat)

plt.plot('x','y',data = lineardata, marker = '^', linestyle = '', label = 'data')
plt.plot(xFit, yFit, color = 'r', lw = 3, linestyle = '--', label = 'Linear fit')
plt.xlabel('x',fontsize = 18)
plt.ylabel('y', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()

# Task is incomplete
