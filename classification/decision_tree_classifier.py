# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:25:37 2018

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
data = pd.read_csv("C:/Users/ayo/Desktop/udemy/DataScience-Python3/data/Social_Network_Ads.csv")

# EXTRACTING THE INDEPENDENT VARIABLES
x = data.iloc[:,2:4].values
# EXTRACTING THE DEPENDENT VARIABLES
y = data.iloc[:,4].values

# split the data into trai and test set
from sklearn.model_selection import train_test_split
x_train ,x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
scale = StandardScaler()
x_train=sc_X.fit_transform(x_train)
x_test=sc_X.fit_transform(x_test)

# using the classifier to the train set
#define the classifier here
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(x_train,y_train)
# predicting the test set
y_pred=classifier.predict(x_test)
y_pred

# visualizing the training set results
# mpl.rcParams["fig.size"]=(10,10)
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min() -1,stop=x_set[:,0].max()+1,step=0.01),
                   np.arange(start=x_set[:,1].min() -1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                              alpha=0.75,cmap=ListedColormap(("red","green")))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
               c=ListedColormap(("red","green"))(i),label=j)
plt.title('classifier training set')
plt.xlabel("age")
plt.ylabel("estimated salary")
plt.legend()
plt.show()

# visualizing the training set results
# mpl.rcParams["fig.size"]=(10,10)
from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min() -1,stop=x_set[:,0].max()+1,step=0.01),
                   np.arange(start=x_set[:,1].min() -1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                              alpha=0.75,cmap=ListedColormap(("red","green")))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
               c=ListedColormap(("red","green"))(i),label=j)
plt.title("classifier'testing set'")
plt.xlabel("age")
plt.ylabel("estimated salary")
plt.legend()
plt.show()

#cm
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
