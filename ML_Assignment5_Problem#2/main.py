# -------------------------------------------------------------------------
# AUTHOR: Joshua Furman
# FILENAME: clustering.py
# SPECIFICATION: uses the clutering alrgorithm using k-means to find best k value
# FOR: CS 4210- Assignment #5
# TIME SPENT: 1 hr
# -----------------------------------------------------------*/

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score
import matplotlib.pyplot as plt

# Read the training data
df_training = pd.read_csv('training_data.csv', sep=',', header=None)
X_training = df_training.values

# Run k-means for different k values
silhouette_scores = []
k_values = range(2, 21)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    silhouette_scores.append(silhouette_score(X_training, kmeans.labels_))

# Find the best k value that maximizes the Silhouette coefficient
best_k = k_values[np.argmax(silhouette_scores)]

# Plot the Silhouette coefficients
plt.plot(k_values, silhouette_scores)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient for each k value')
plt.show()

# Perform k-means clustering with the best k value
kmeans = KMeans(n_clusters=best_k, random_state=0)
kmeans.fit(X_training)

# Read the testing data (classes)
df_testing = pd.read_csv('testing_data.csv', sep=',', header=None)
labels_true = np.array(df_testing.values).reshape(1, -1)[0]

# Obtain the predicted labels from the best k clustering
labels_pred = kmeans.labels_

# Calculate the Homogeneity score
homogeneity = homogeneity_score(labels_true, labels_pred)
print("Homogeneity Score:", homogeneity)
