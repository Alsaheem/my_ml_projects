# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:30:13 2018

@author: ayo
"""

 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("C:/Users/ayo/Desktop/udemy/DataScience-Python3/data/Market_Basket_Optimisation.csv",header=None)
transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i,j]) for j in range(20)])
    
#training priori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

 #visualizing the results
results = list(rules)