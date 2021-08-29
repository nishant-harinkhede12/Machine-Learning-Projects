import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/akjadon/HH/master/Python/DS_tutorials_all/Data/churn.csv.txt'

data = pd.read_csv(url,sep=',', parse_dates=['last_trip_date','signup_date'])
data.info()
data['last_trip_date'].max()
#-------------------------------------------------------------------------------------------

import datetime
cutoff = data['last_trip_date'].max() - datetime.timedelta(30,0,0)
data['churn'] = (data['last_trip_date'] < cutoff).astype(int)
data.head()
#-------------------------------------------------------------------------------------------

# categorical = [a for a in data.columns if data[a].dtype=='O']
# numerical = [a for a in data.columns if data[a].dtype=='1']

categorical_col = data.select_dtypes('object').columns
numerical_col = data.select_dtypes('float64').columns
#-------------------------------------------------------------------------------------------
data.isnull().sum()
#-------------------------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pipeline_num = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scaling',StandardScaler())])

pipeline_cat = Pipeline(steps=[('imputer',SimpleImputer(strategy='constant', fill_value='missing')),
                               ('encoding',OneHotEncoder(handle_unknown='ignore'))])

preprocesser = ColumnTransformer(
    transformers=[('numerical',pipeline_num,numerical_col),
                  ('categorical',pipeline_cat,categorical_col)])
#-------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pipeline = Pipeline(steps=[('preprocessor',preprocesser),
                ('classifier',RandomForestClassifier(n_estimators=10))])

x_train, x_test, y_train, y_test = train_test_split(data,data['churn'],test_size=0.2, random_state = 0)

pipeline.fit(x_train,y_train)

pipeline.score(x_test,y_test)
