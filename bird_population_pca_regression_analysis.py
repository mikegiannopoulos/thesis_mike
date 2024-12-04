"""
This script applies Principal Component Analysis (PCA) to the climate dataset and uses the
transformed data to train a Linear Regression model to predict the individual count of birds.

The script starts by loading the dataset, replacing -1 values with NaN, and imputing missing values.
Then, it selects the relevant columns for PCA, standardizes the data, and fits a PCA model to
the data. The explained variance ratio is plotted to decide the optimal number of components to
retain. The PCA model is then refit with the optimal number of components, and the transformed
data is split into training and testing sets. A Linear Regression model is fit to the training
data, and its performance is evaluated on the test data.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your data
final_bird_data = pd.read_csv('/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/data_for_correlation.csv')

# Replace -1 with NaN and impute missing values
filtered_data = final_bird_data.replace(-1, np.nan)
filtered_data['air_pressure_values'].fillna(filtered_data['air_pressure_values'].mean(), inplace=True)
filtered_data['air_temperature_values'].fillna(filtered_data['air_temperature_values'].mean(), inplace=True)
filtered_data['wind_values'].fillna(filtered_data['wind_values'].mean(), inplace=True)
filtered_data['sea_temp_values'].fillna(filtered_data['sea_temp_values'].mean(), inplace=True)
filtered_data['seawater_level_values'].fillna(filtered_data['seawater_level_values'].mean(), inplace=True)
filtered_data['wave_height_values'].fillna(filtered_data['wave_height_values'].mean(), inplace=True)

# Select relevant columns for PCA
X = filtered_data[['air_pressure_values', 'air_temperature_values', 
                   'wind_values', 'sea_temp_values', 'seawater_level_values', 
                   'wave_height_values']]
y = filtered_data['individualCount']

# Standardize the data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Fit PCA with all components and analyze the explained variance
pca = PCA(n_components=None)  # Keep all components for now
X_pca = pca.fit_transform(X_scaled)

# Step 2: Plot the explained variance ratio to decide the number of components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.tight_layout()

# Save the plot to results folder
results_folder = '/home/michael/Education/UoG/Earth Science Master/Thesis/results/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
plt.savefig(os.path.join(results_folder, "pca_explained_variance_plot.png"), dpi=300)
plt.show()

# Step 3: Fit PCA with optimal number of components (e.g., choose components explaining 90% variance)
pca_optimal = PCA(n_components=0.90)  # Retain 90% of the variance
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Check the shape of the transformed dataset
print(f"Original shape: {X.shape}, Transformed shape: {X_pca_optimal.shape}")

# Split the transformed data into training and testing sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_optimal, y, test_size=0.2, random_state=42)

# Fit a Linear Regression model on the PCA-transformed data
model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train_pca)

# Predict on the test set
y_pred_pca = model_pca.predict(X_test_pca)

# Evaluate the model
mse_pca = mean_squared_error(y_test_pca, y_pred_pca)
r2_pca = r2_score(y_test_pca, y_pred_pca)

print(f"PCA Mean Squared Error: {mse_pca}")
print(f"PCA R-squared: {r2_pca}")

# Save the PCA model results to a CSV file
results = pd.DataFrame({
    'Model': ['Linear Regression with PCA'],
    'Mean Squared Error': [mse_pca],
    'R-squared': [r2_pca]
})

# Save the results to the results folder
results_file_path = os.path.join(results_folder, 'pca_regression_model_performance.csv')
results.to_csv(results_file_path, index=False)

print(f"Results saved to {results_file_path}")
