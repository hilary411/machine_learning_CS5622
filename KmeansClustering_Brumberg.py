# K Means Clustering and Silhoutte Score

import pandas as pd
import numpy as np

import sklearn

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import os as os
import matplotlib.pyplot as plt

import seaborn as sns


Bahia = pd.read_csv('BahiaSubsample.csv')
#print(Bahia)

# Just keep columns of interest for comparing Animal Production and Percent Area Restored
Bahia_restore = Bahia.drop(Bahia.columns[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10]], axis=1)
print(Bahia_restore)


# In[15]:


#Calculating Silhoutte Score for various values of k

cluster = [2,3,4,6,7,8,9,10]
silhouette_scores = []


for i in cluster:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(Bahia_restore.iloc[:, :-1])
    silhouette_avg = silhouette_score(Bahia_restore.iloc[:, :-1], cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(silhouette_avg)


# Silhouette Score graph
plt.figure(figsize=(8, 6))
plt.plot(range(2, 10), silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()


# Find the number of clusters with the highest Silhouette Score
best_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print('The optimal number of clusters according to Silhouette Score for Bahia Restore is', best_num_clusters)


# In[28]:


# K-Means clustering for 2 clusters, the best number of clusters as determined by Silhouette Score
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(Bahia_restore.iloc[:, :-1])
Bahia_restore['cluster'] = cluster_labels
print(Bahia_restore.head())

# Plot the 2 clusters
cluster_0 = Bahia_restore[Bahia_restore['cluster'] == 0]
cluster_1 = Bahia_restore[Bahia_restore['cluster'] == 1]

plt.scatter(cluster_0.iloc[:, 0], cluster_0.iloc[:, 1], s=50, c='green', label='Cluster 1')
plt.scatter(cluster_1.iloc[:, 0], cluster_1.iloc[:, 1], s=50, c='blue', label='Cluster 2')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*', label='Centroids')

plt.title('K-means Clustering for k=2')
plt.xlabel('Animal production')
plt.ylabel('Percent restored')
plt.legend()
plt.show()


# In[27]:


# Comparing K-Means clustering with 2, 3, and 4 clusters


colors = ['green', 'blue', 'yellow', 'grey']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))  

#iterate through each K value. Start by calculating K means
for index, k in enumerate([2,3,4]):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(Bahia_restore.iloc[:, :-1])
    Bahia_restore['cluster'] = cluster_labels

    # Plot clusters and centroids
    for cluster_num in range(k):
        cluster_data = Bahia_restore[Bahia_restore['cluster'] == cluster_num]
        axes[index].scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], s=50, c=colors[cluster_num], label=f'Cluster {cluster_num+1}')
    axes[index].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*', label='Centroids')
    
    # Figure design
    axes[index].set_title(f'K={k}')
    axes[index].set_xlabel('Animal production')
    if index == 0: 
        axes[index].set_ylabel('Percent restored')
    axes[index].legend()
plt.tight_layout()
plt.show()

