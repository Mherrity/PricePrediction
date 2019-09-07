
# coding: utf-8

# In[17]:


import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

#matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style


# In[18]:


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2019, 1, 11)

df = web.DataReader("GLUU", 'yahoo', start, end)
df.tail()


# In[19]:


dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0


# In[20]:


dfreg.head()


# In[21]:


import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm


# In[22]:


# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

print('Dimension of X',X.shape)
print('Dimension of y',y.shape)


# In[23]:


# Separation of training and testing of model by cross validation train test split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


# In[43]:


from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from sklearn.pipeline import make_pipeline


# In[44]:


#Lasso regression
clflas = linear_model.Lasso(alpha=0.1)
clflas.fit(X_train, y_train)

#LARS Lasso regression
clfrid = linear_model.Ridge(alpha=.5)
clfrid.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)


# In[49]:


confidencelas = clflas.score(X_test, y_test)
confidencerid = clfrid.score(X_test, y_test)
confidenceknn = clfknn.score(X_test, y_test)

print("The Lasso regression confidence is ",confidencelas)
print("The Ridge regression confidence is ",confidencerid)
print("The KNN regression confidence is ",confidenceknn)


# In[53]:


# Printing the forecast
forecast_set = clfrid.predict(X_lately)
dfreg['Forecast'] = np.nan
print(forecast_set, confidencerid, forecast_out)


# In[54]:


last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]


# In[55]:


dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

