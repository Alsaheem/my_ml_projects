#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import sklearn

#reading the dataset
data = pd.read_csv("C:/Users/ayo/Desktop/MY_udemy_courses/datascience/datasets/Salary_Data.csv")
#data
X = data.iloc[:,:-1].values
y = data.iloc[:,1]

#splitting the data into training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
"""

#fitting simple linear regression into our model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#pedicting the values of the test set
y_pred=regressor.predict(X_test)

#visualizing the taining set results
plt.scatter(X_train,y_train,c="r")
plt.plot(X_train,regressor.predict(X_train),c="b")
plt.ylabel("SALARY")
plt.xlabel("AGE")
plt.title("graph of salary against years of expirience")
plt.show()

#visualizing the testing set results
plt.scatter(X_test,y_test,c="r")
plt.plot(X_train,regressor.predict(X_train),c="b")
plt.ylabel("SALARY")
plt.xlabel("AGE")
plt.title("graph of salary against years of expirience")
plt.show()