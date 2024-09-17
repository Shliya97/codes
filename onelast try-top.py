import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer

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

plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importances[::-1], color='darkblue')
plt.xlabel('Importance Score')
plt.ylabel('CpG sites')
plt.title('Feature Importance by Random Forest')
plt.show()

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

# Optionally, save the selected top N features and their importances
top_features_df = pd.DataFrame({
    'Feature': top_features,
    'Importance': top_importances
})
top_features_df.to_csv(f'top_{N}_features_rf.csv', index=False)

print(f"Top {N} features saved to 'top_{N}_features_rf.csv'")
# Calculate correlation between each CpG site and age
correlations = []
for cpg_site in data.columns[:-1]:  # Exclude 'age' column
    corr = data[cpg_site].corr(data['age'])
    correlations.append((cpg_site, abs(corr)))

# Sort CpG sites by absolute correlation value
correlations.sort(key=lambda x: x[1], reverse=True)

# Select top CpG sites with the highest absolute correlation values
# Select top 10 for example
top_cpg_sites = [cpg for cpg, _ in correlations[:10]]
top_cpg_data = data[top_cpg_sites].copy()
top_cpg_data['age'] = data['age'].values

# Save the selected data to a new CSV file
top_cpg_data.to_csv('top_cpg_sites_LT.csv', index=False)

# Prepare the data
X = data.drop(columns=['age'])
y = data['age']
# Impute missing values in X using the mean of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_imputed, y)

# Get feature importances
importances = rf.feature_importances_

# Select top CpG sites
indices = np.argsort(importances)[-10:]  # Select top 10 for example
top_cpg_sites = X.columns[indices]
top_cpg_data = data[top_cpg_sites].copy()
top_cpg_data['age'] = data['age'].values

# Save the selected data to a new CSV file
top_cpg_data.to_csv('top_cpg_sites_rf_LT.csv', index=False)

# Train a Lasso model
lasso = Lasso(alpha=0.01, max_iter=10000)  # Adjust alpha as needed
lasso.fit(X_imputed, y)

# Get feature coefficients
coefficients = lasso.coef_

# Select top CpG sites
indices = np.argsort(np.abs(coefficients))[-10:]
top_cpg_sites = X.columns[indices]
top_cpg_data = data[top_cpg_sites].copy()
top_cpg_data['age'] = data['age'].values

# Save the selected data to a new CSV file
top_cpg_data.to_csv('top_cpg_sites_lasso_LT.csv', index=False)
