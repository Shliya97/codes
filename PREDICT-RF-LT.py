#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:19:53 2024

@author: laisha
"""

# Set the style for the plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
sns.set(style="whitegrid")


# Set the style for the plots
sns.set(style="whitegrid")
# Load the dataset
data = pd.read_csv('top_cpg_sites_rf_LT.csv', index_col=0)


# Display the first few rows of the dataset
print(data.head())
# Remove samples with missing values
data = data.dropna()

# Normalize the methylation data (excluding the age column)
methylation_data = data.drop(columns=['age'])
scaler = StandardScaler()
normalized_data = scaler.fit_transform(methylation_data)

# Convert normalized data back to a DataFrame
normalized_data = pd.DataFrame(normalized_data,
                               columns=methylation_data.columns,
                               index=methylation_data.index)

# Add the age column back to the normalized data
normalized_data['age'] = data['age'].values

# Display the first few rows of the normalized data
print(normalized_data.head())


# Define a function to group ages into a range
def group_by_decades(age):
    lower_bound = (age // 10) * 10
    upper_bound = lower_bound + 9
    return f"{lower_bound}-{upper_bound}"


# Apply the function to create a new column 'age_group'
normalized_data['age_group'] = normalized_data['age'].apply(group_by_decades)

# Save the processed data to a new CSV file
normalized_data.to_csv('Processed-DNA-data_rf_LT.csv', index=False)

# Display the first few rows of the data with age groups
print(normalized_data.head())

# Define and sort age groups in the correct order
age_groups_ordered = sorted(
    normalized_data['age_group'].unique(), key=lambda x: int(x.split('-')[0]))

# Convert 'age_group' to ordered categorical type with the correct order
normalized_data['age_group'] = pd.Categorical(
    normalized_data['age_group'],
    categories=age_groups_ordered,
    ordered=True
)

# Plot the distribution of age groups with correct order
plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', data=normalized_data, order=age_groups_ordered)
plt.title('Age Group Distribution')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

# Box plot of methylation data for a specific CpG site grouped by age group
cpG_site = methylation_data.columns[0]  # Select the first CpG site for demo
plt.figure(figsize=(12, 8))
sns.boxplot(x='age_group', y=cpG_site, data=normalized_data)
plt.title(f'Box Plot of {cpG_site} by Age Group')
plt.xlabel('Age Group')
plt.ylabel(f'{cpG_site} Methylation Level')
plt.show()

# Plot DNA methylation percentage on y-axis and age on x-axis for each CpG site
cpg_sites = data.columns[:-1]  # Exclude the 'age' column

# Plotting with linear regression lines

for cpg in cpg_sites:
    plt.figure(figsize=(10, 6))

    # Linear regression plot using Seaborn
    sns.regplot(x='age', y=cpg, data=data, ci=None, scatter_kws={'alpha': 0.6})

    plt.xlabel('Age')
    plt.ylabel('DNA Methylation level')
    plt.title(f'Regression Plot of {cpg}')
    plt.show()

# Load the preprocessed dataset
data = pd.read_csv('Processed-DNA-data_rf_LT.csv')

# Display the first few rows of the dataset
print(data.head())

# Separate features (CpG sites) and target variable (age)
X = data.drop(columns=['age', 'age_group'])
y = data['age']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Display the shapes of the training and testing sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Train the model using the training set
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_model.predict(X_test)
# Define the hyperparameters and their values to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=3, n_jobs=-1, scoring='neg_mean_absolute_error',
                           verbose=2)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best hyperparameters: ", best_params)

# Train the Random Forest model with the best hyperparameters
rf_model_optimized = RandomForestRegressor(**best_params, random_state=42)
rf_model_optimized.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_optimized = rf_model_optimized.predict(X_test)
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred_optimized)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
r2 = r2_score(y_test, y_pred_optimized)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Plot actual vs. predicted ages
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_optimized, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--',
         lw=2)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs Predicted Age')
plt.show()
