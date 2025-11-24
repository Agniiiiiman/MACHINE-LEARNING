'''Hierarchical Clustering is an unsupervised learning technique that
 builds a hierarchy (tree-like structure) of clusters without requiring you to predefine the number of clusters (unlike K-Means).'''
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Sample Data
X = np.array([[1,2], [2,3], [5,6], [6,7], [3,2], [8,9]])

# Plot dendrogram using linkage
linked = linkage(X, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Dendrogram")
plt.show()

# Perform Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = cluster.fit_predict(X)

print("Cluster labels:", labels)
