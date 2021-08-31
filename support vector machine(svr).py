import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'E:\Phase 1\Machine Learning\Task\Support vector mc\Position_Salaries.csv')

x = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

from sklearn.svm import SVR

reg = SVR(kernel = 'rbf')
reg.fit(x,y)

ypred = reg.predict([[6.5]])

plt.scatter(x, y, color = 'red')
plt.plot(x, reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

###########################################################################################

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

x1 = sc_x.fit_transform(x)
y1 = np.squeeze(sc_y.fit_transform(y.reshape(-1,1))) 

from sklearn.svm import SVR

reg = SVR(kernel = 'rbf')
reg.fit(x,y)

ypred = reg.predict([[6.5]])

y_pred = sc_y.inverse_transform(reg.predict(sc_x.transform(np.array([[6.5]]))))

plt.scatter(x, y, color = 'red')
plt.plot(x, reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(x), max(x), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()