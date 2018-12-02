# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 18:07:05 2018

@author: ayo
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from thompson_sampling_algorithm import *

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

ads, total_result = thompson_sampling(dataset.values)
total_result

#visualizing
plt.hist(ads)
plt.title('Thompson sampling algorithm on ads strategy')
plt.xlabel('Different versions of ads')
plt.ylabel('Number of times ads were selected')
plt.show()



