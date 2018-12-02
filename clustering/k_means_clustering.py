# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:48:18 2018

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

#finding the number of clusters to use in kmeans
from sklearn.cluster import KMeans
#wcss = within clusters sum of squares
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title(" the elbow method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()

#fitting the kmeans algorithmm to our dataset
kmeans=KMeans(n_clusters=5,init="k-means++",max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="r",label="cluster 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c="b",label="cluster 2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c="g",label="cluster 3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c="y",label="cluster 4")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c="k",label="cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="c",label="centroids")
plt.title("clusters of  clients")
plt.xlabel("annual income")
plt.ylabel("spending score 1-100")
plt.legend()
plt.show()