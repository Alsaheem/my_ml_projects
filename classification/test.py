# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:11:58 2018

@author: ayo
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import sklearn

# IMPORTING THE DATASET
data = pd.read_excel("C:/Users/ayo/Desktop/sales_and_alertss.xlsx")

# EXTRACTING THE INDEPENDENT VARIABLES
x = data.iloc[:,3:5].values
# EXTRACTING THE DEPENDENT VARIABLES
y = data.iloc[:,2].values

#encoding categorical vaiables
#from sklearn.preprocessing import LabelEncoder
#dependent variable
#labelencoder_y=LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# split the data into trai and test set
from sklearn.model_selection import train_test_split
x_train ,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# feature scaling
#sc_X=StandardScaler()
#scale = StandardScaler()
#x_train=sc_X.fit_transform(x_train)
#x_test=sc_X.fit_transform(x_test)

# using the classifier to the train set
#define the classifier here
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
classifier.fit(x_train,y_train)

# predicting the test set
y_pred=classifier.predict(x_test)
y_pred 
