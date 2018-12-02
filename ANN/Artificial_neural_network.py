# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 08:10:19 2018

@author: ayo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# IMPORTING THE DATASET
data = pd.read_csv("Churn_Modelling.csv")

# EXTRACTING THE INDEPENDENT VARIABLES
X = data.iloc[:,3:13].values
# EXTRACTING THE DEPENDENT VARIABLES
y = data.iloc[:,13].values

#encoding categorical vaiables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
#independent variable
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# split the data into trai and test set
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
scale = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

# part 2
#importing keras library and package
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier=Sequential()

#adding the input layer and the first hidden layer
classifier.add(Dense(6,input_shape=(11,),kernel_initializer="uniform",activation="relu"))

#adding the second hidden layer
classifier.add(Dense(6,kernel_initializer="uniform",activation="relu"))

#adding the third hidden layer
classifier.add(Dense(6,kernel_initializer="uniform",activation="relu"))

#adding the output layer
classifier.add(Dense(1,kernel_initializer="uniform",activation="sigmoid"))
"""
if your dependent/output variable has more than 2 categories, we use (activation="softmax")
i.e more than 0 and 1
"""
#compiling the ann
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
"""
if your dependent/output variable has a binary outcome then the
logarithmic loss = "binary_crossentropy"
if your dependent/output variable has more than 2 categories, the
logarithmic loss= "categorical_crossentropy"
"""

#fitting the classifier to the train set
classifier.fit(X_train,y_train,batch_size=10,epochs=50)
#define the classifier here

# predicting the test set
y_pred=classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#making the prediction and evaluating the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#.to_csv()