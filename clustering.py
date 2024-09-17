#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:30:46 2024

@author: laisha
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load the dataset
data = pd.read_csv('one last try with age.csv', index_col=0)

# Prepare the data
X = data.drop(columns=['age'])
y = data['age']

# Impute missing values in X using the mean of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
# Ensure DataFrame with column names
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_imputed_df, y)

# Get feature importances
importances = rf.feature_importances_
# Sort features by importance
sorted_indices = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_indices]
sorted_features = X.columns[sorted_indices]

# Optionally, select top N features based on the plot
N = 10  # Choose N based on your needs or the plot
top_features = sorted_features[:N]  # Select top N features
top_importances = sorted_importances[:N]


# Subset your data with the top N features
X_top = X[top_features]

# Fit a new imputer on the subset of top features
imputer_top = SimpleImputer(strategy='mean')
X_top_imputed = imputer_top.fit_transform(X_top)
# DataFrame with top feature names
X_top_imputed_df = pd.DataFrame(X_top_imputed, columns=top_features)

# Example: Train a new Random Forest model with the top N features
rf_top = RandomForestRegressor(n_estimators=100, random_state=42)
rf_top.fit(X_top_imputed_df, y)

# Perform clustering using KMeans on the top N features
kmeans = KMeans(n_clusters=3, random_state=42)  # Choose the number of clusters
clusters = kmeans.fit_predict(X_top_imputed_df)

# Add cluster labels to the DataFrame for analysis
X_top_imputed_df['Cluster'] = clusters

# Visualize clustering
plt.figure(figsize=(8, 6))
plt.scatter(X_top_imputed_df.iloc[:, 0],
            X_top_imputed_df.iloc[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Top Feature 1(cg26015888)')
plt.ylabel('Top Feature 2(cg26398791)')
plt.title('Clustering of Samples Based on Top Features')
plt.show()
