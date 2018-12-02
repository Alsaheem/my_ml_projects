# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 08:14:58 2018

@author: ayo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import sklearn

#reading the dataset
data = pd.read_csv("C:/Users/ayo/Desktop/MY_udemy_courses/datascience/datasets/50_Startups.csv")
#data
X = data.iloc[:,:-1].values
y = data.iloc[:,4]

#encoding categorical vaiables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
#independent variable
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X= X[:,1:]

#splitting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#fitting simple linear regression into our model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


#pedicting the values of the test set
y_pred=regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() #we see that the r%d spent is a powerful predictor of the profit
