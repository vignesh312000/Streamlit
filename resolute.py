# -*- coding: utf-8 -*-
"""resolute.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17y8hJ3cYcDeSHi-I_4vdx8bPEmc67zAp
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

drive="/content/drive/MyDrive/train.xlsx"
df=pd.read_excel(drive)

df.head()

"""As far the Clustering technique is a Unsupervised learning so the target label is not needed so the process involves the dropping the variable "target"."""

unique_targets = df['target'].nunique()

print("Unique targets:", unique_targets)

data=df.drop(['target'],axis=1)

# data.head()

"""Detecting Outliers

"""

plt.figure(figsize=(12, 8))
sns.boxplot(data=data, orient="v", palette="Set2")
plt.title("Box Plot of Features to Identify Outliers")
plt.ylabel("Feature Values")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.show()

from scipy.stats import zscore
z_scores = zscore(data)
z_scores

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]

data.fillna(data.mean(), inplace=True)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

# Get cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the dataframe
data['cluster'] = cluster_labels

# Evaluate the quality of clusters using silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

"""Score of 0.218 suggests that there is a moderate degree of separation between the clusters, indicating that the clustering algorithm has produced reasonably distinct clusters."""

# Interpret the clusters and analyze the characteristics of each cluster
cluster_analysis = df.groupby('cluster').mean()
print(cluster_analysis)

"""**Cluster 0:**
- **General Trend**: This cluster has moderate negative values across most features.
- **Key Features**:
  - Features T1 to T18 have relatively moderate negative values.
- **Observations**:
  - The values in this cluster are not extremely low but are still negative overall.
  - There is no strong trend or pattern apparent in the values of the features in this cluster.

**Cluster 1:**
- **General Trend**: This cluster has the lowest negative values across most features.
- **Key Features**:
  - Features T1 to T18 have the lowest mean values compared to the other clusters.
- **Observations**:
  - The values in this cluster are consistently low and represent the lowest measurements among all clusters.
  - This cluster stands out as having the most negative values across all features.

**Cluster 2:**
- **General Trend**: This cluster has relatively high negative values across most features.
- **Key Features**:
  - Features T1 to T18 have higher mean values compared to Cluster 0 but lower compared to Cluster 1.
- **Observations**:
  - The values in this cluster are higher compared to Cluster 0 but still negative overall.
  - There is a noticeable separation from Cluster 1, indicating that this group exhibits higher measurements across the features compared to Cluster 1.

Overall, the clusters exhibit different levels of negative values across the features, with Cluster 1 showing the lowest values and standing out as a distinct group. These observations provide insight into the characteristics of each cluster and can help in understanding the differences between them.
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.5, label='Data Points')

# Plot centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', c='red', s=200, label='Centroids')

plt.title('K-means Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# Calculate Davies-Bouldin index
db_index = davies_bouldin_score(X_scaled, cluster_labels)
print(f"Davies-Bouldin Index: {db_index}")

# Calculate Calinski-Harabasz index
ch_index = calinski_harabasz_score(X_scaled, cluster_labels)
print(f"Calinski-Harabasz Index: {ch_index}")

"""Davies-Bouldin Index: The value of approximately 1.44 indicates a moderate level of separation between the clusters. Lower values suggest better clustering, indicating that the clusters are compact and well-separated.

Calinski-Harabasz Index: The value of approximately 12880.10 is relatively high, indicating that the clusters are well-separated and dense. Higher values indicate better clustering, suggesting that the clusters are distinct from each other.

Cross_validation
"""

from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

# Assuming X is your feature matrix
X = X_scaled

# Define the number of folds for cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize a list to store silhouette scores for each fold
silhouette_scores = []

# Perform cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]

    # Fit K-means clustering model on the training set
    kmeans = KMeans(n_clusters=3)  # Example: using KMeans with 3 clusters
    kmeans.fit(X_train)

    # Predict cluster labels for the test set
    cluster_labels = kmeans.predict(X_test)

    # Calculate silhouette score for the current fold
    silhouette_avg = silhouette_score(X_test, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Calculate the average silhouette score across all folds
average_silhouette_score = np.mean(silhouette_scores)
print("Average Silhouette Score:", average_silhouette_score)