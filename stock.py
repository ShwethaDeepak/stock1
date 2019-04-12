#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:09:21 2018

@author: swetu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('data_stocks.csv')

#=========Determining the number of cluster using Elbow method===========
print("Determining the number of cluster using Elbow method")

ks = range(1,10)
inertias = []
for k in ks:
    model = KMeans(n_clusters = k)
    model.fit(df)
    inertias.append(model.inertia_)
plt.plot(ks,inertias,'-o')
plt.xlabel('no of cluster,k')
plt.ylabel('Inertias')

df_data = df.iloc[:,1:-1].values

kmeans = KMeans(n_clusters=4)

labels = kmeans.fit_predict(df_data)

df['labels'] = labels
df.sort_values('labels')

#===Number of unique patterns=====
print("Number of unique patterns")

df['labels'].unique()

#============stocks moving together and stocks are different from each other===
print("stocks moving together and stocks are different from each other")
print("stocks apparently similar in performance")
df_cat0= df.loc[df['labels']==0]
df_cat1= df.loc[df['labels']==1]
df_cat2=df.loc[df['labels']==2]
df_cat3=df.loc[df['labels']==3]





