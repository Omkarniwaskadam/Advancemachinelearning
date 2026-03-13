from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Generate sample data (3 clusters)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 2. Initialize and fit the K-Medoids model
# We set n_clusters=3 to match our data
kmedoids = KMedoids(n_clusters=3, random_state=0, metric='manhattan').fit(X)

# 3. Retrieve results
labels = kmedoids.labels_
medoids = kmedoids.cluster_centers_

# 4. Visualize the clusters and their medoids
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5, label='Data Points')
plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='X', s=200, label='Medoids')
plt.title('K-Medoids Clustering')
plt.legend()
plt.show()

print(f"Medoid coordinates:\n{medoids}")
