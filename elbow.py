#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:38:48 2024

@author: laisha
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

# Load the dataset
data = pd.read_csv('one last try with age.csv', index_col=0)

# Step 2: Optional - Standardize the data (normalize each feature to have mean=0 and variance=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: Calculate inertia for different values of k
inertias = []
# You can choose k values from 1 to 10 or any range you prefer
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertias.append(kmeans.inertia_)

# Step 4: Find the elbow point using the "maximum curvature" method
# Normalize the k values and inertia for better precision
k_values_normalized = np.array(k_values)
inertias_normalized = np.array(inertias)

# Create line vector from the first to the last point
point_1 = np.array([k_values_normalized[0], inertias_normalized[0]])
point_2 = np.array([k_values_normalized[-1], inertias_normalized[-1]])

# Calculate the distance from each point to the line
distances = []
for i in range(len(k_values_normalized)):
    point = np.array([k_values_normalized[i], inertias_normalized[i]])
    distance = np.abs(np.cross(point_2 - point_1, point_1 -
                      point)) / np.linalg.norm(point_2 - point_1)
    distances.append(distance)

# Find the index of the elbow point (maximum distance)
elbow_index = np.argmax(distances)
elbow_k = k_values[elbow_index]

# Step 5: Plot the Elbow Method results and mark the elbow point
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertias, 'bo-', markersize=8, label='Inertia')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.xticks(k_values)
plt.grid(True)

# Mark the elbow point
plt.axvline(x=elbow_k, color='r', linestyle='--',
            label=f'Elbow at k={elbow_k}')
plt.scatter(elbow_k, inertias[elbow_index],
            color='red', s=150, zorder=5, label='Elbow Point')

# Add legend
plt.legend()

# Show the plot
plt.show()

print(
    f"The optimal number of clusters (k) using the elbow method is: {elbow_k}")
