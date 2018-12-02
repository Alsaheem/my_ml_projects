# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:33:15 2018

@author: ayo
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import sklearn

#reading the dataset
data = pd.read_csv("C:/Users/ayo/Desktop/udemy/DataScience-Python3/data/Position_Salaries.csv")
#data
X = data.iloc[:,[1]].values
y = data.iloc[:,2].values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#visualizing the linear regressor results
plt.scatter(X,y,c="r")
plt.plot(X,lin_reg.predict(X),c="b")
plt.ylabel("SALARY")
plt.xlabel("AGE")
plt.title("graph of salary against years of expirience")
plt.show()

#visualizing the polynimial regressor results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,c="r")
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),c="b")
plt.ylabel("SALARY")
plt.xlabel("AGE")
plt.title("graph of salary against years of expirience")
plt.show()

#predicting a new salary with linear regression
salary_lin = lin_reg.predict(6.5)
print("using linear regression the salary is " , salary_lin)

#predicting a new salary with linear regression
salary_poly = lin_reg2.predict(poly_reg.fit_transform(6.5))
print("using polynomial regression the salary is " , salary_poly)