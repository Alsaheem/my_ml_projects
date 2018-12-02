# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:06:52 2018

@author: ayo
"""

#regressiion template

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

"""
#splitting the data into training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_y=StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""
#fitting the regressor model into the dataset
#create your regressor

#predicting a new salary with  regression
y_pred= regressor.predict(6.5)


#visualizing the p regressor results(for higher resolution and smoother curves)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,c="r")
plt.plot(X_grid,regressor.predict(X_grid),c="b")
plt.ylabel("SALARY")
plt.xlabel("posiion")
plt.title("regression model")
plt.show()