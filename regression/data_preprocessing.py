# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:38:20 2018

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
data = pd.read_excel("C:/Users/ayo/Desktop/udemy/DataScience-Python3/data/data.xlsx")
#data
X = data.iloc[:,:-1].values
y = data.iloc[:,3]

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


#encoding categorical vaiables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
#independent variable
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
#dependent variable
labelencoder_y=LabelEncoder()
y = labelencoder_X.fit_transform(y)

#splitting the data into training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)