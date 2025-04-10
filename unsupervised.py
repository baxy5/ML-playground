"""
Unsupervised learning: The model works with unlabeled data to find patterns or structure.

Models:
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage


# -------------------- K-Means Clustering -----------------------------------------
# Generate data
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=10, random_state=42)

plt.scatter(X[:, 0], X[:, 1], s=50, cmap="viridis")
plt.title("Raw Data (Before Clustering)")
plt.show()

# Fit the K-Means model
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Get the cluster assignments and centroids
y_kmeans = kmeans.predict(X)
centroids = kmeans.cluster_centers_

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis")
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    s=200,
    alpha=0.75,
    marker="X",
    label="Centroids",
)
plt.title("K-Means Clustering (k=3)")
plt.legend()
plt.show()

# -------------------- Hierarchical Clustering -----------------------------------------
X, _ = make_blobs(n_samples=150, centers=4, cluster_std=1.0, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 6))
dendrogram(linked, orientation="top", distance_sort="ascending", show_leaf_counts=True)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

model = AgglomerativeClustering(n_clusters=4)
labels = model.fit_predict(X_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="tab10", s=50)
plt.title("Hierarchical Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
