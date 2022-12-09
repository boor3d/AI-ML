"""
PROGRAM WILL NOT RUN DUE TO 403 FORBIDDEN ERROR
MUST HAVE PERMISSIONS ON CODECADEMY SERVER
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print (df.info())
print (df.head())

prod_per_year = df.groupby("year").totalprod.mean().reset_index()
# print (prod_per_year)

X = prod_per_year['year']
X = X.values.reshape(-1,1)

y = prod_per_year['totalprod']

plt.scatter(X, y)
plt.show()

regr = LinearRegression()
regr.fit(X, y)

print(regr.coef_)

print(regr.intercept_)


y_predict = regr.predict(X)

plt.plot(X, y_predict)
plt.show()

X_future = np.array(range(2013,2050))

X_future = X_future.reshape(-1,1)

# Predict the future production with X_future years as X-input
future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict)

plt.show()