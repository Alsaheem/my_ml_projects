# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 08:20:21 2018

@author: ayo
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# IMPORTING THE DATASET
dataset = pd.read_csv("C:/Users/ayo/Desktop/udemy/DataScience-Python3/data/Mall_Customers.csv")

X = dataset.iloc[:, [3, 4]].values

#using the dendogram t0 find the optimzl number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method="ward"))
plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('Euclidean distances')
plt.show()

#fitting the heirachicl clustering into the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_hc = hc.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c="r",label="careful")
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c="b",label="standard")
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c="g",label="extravagant(TARGET)")
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c="y",label="careless(fake life)")
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c="k",label="sensible")
plt.title("clusters of  clients")
plt.xlabel("annual income")
plt.ylabel("spending score 1-100")
plt.legend()
plt.show()