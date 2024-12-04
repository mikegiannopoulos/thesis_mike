"""
This script performs exploratory factor analysis (EFA) on a dataset of climate variables
and uses the resulting factors to predict the count of individual birds.

The script first loads the data, replaces missing values with the mean, and selects the
relevant columns for EFA. It then standardizes the data and performs EFA with 3 factors.
The factor loadings are printed and the model fit is evaluated. The script then splits the
EFA-transformed data into training and testing sets and fits a Linear Regression model.
The model is evaluated on the test set and the mean squared error and R-squared values
are printed.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

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

# Select relevant columns for EFA
X = filtered_data[['air_pressure_values', 'air_temperature_values', 
                   'wind_values', 'sea_temp_values', 'seawater_level_values', 
                   'wave_height_values']]
y = filtered_data['individualCount']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Perform exploratory factor analysis (EFA)
fa = FactorAnalyzer(n_factors=3, rotation='varimax')  # We'll start by extracting 3 factors
fa.fit(X_scaled)

# Step 2: Get the factor loadings
loadings = fa.loadings_

# Step 3: Save and print the factor loadings
factor_loadings_df = pd.DataFrame(loadings, columns=['Factor 1', 'Factor 2', 'Factor 3'], index=X.columns)
print("Factor Loadings:\n", factor_loadings_df)

# Create results folder if it doesn't exist
results_folder = '/home/michael/Education/UoG/Earth Science Master/Thesis/results/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Save the factor loadings to CSV
factor_loadings_file_path = os.path.join(results_folder, 'efa_factor_loadings.csv')
factor_loadings_df.to_csv(factor_loadings_file_path)
print(f"Factor loadings saved to {factor_loadings_file_path}")

# Step 4: Evaluate the model fit (optional: check the model adequacy)
# Use the transformed data (EFA scores) for further analysis if needed
efa_scores = fa.transform(X_scaled)

# Split the EFA-transformed data into training and testing sets
X_train_efa, X_test_efa, y_train_efa, y_test_efa = train_test_split(efa_scores, y, test_size=0.2, random_state=42)

# Fit a Linear Regression model on the EFA-transformed data
model_efa = LinearRegression()
model_efa.fit(X_train_efa, y_train_efa)

# Predict on the test set
y_pred_efa = model_efa.predict(X_test_efa)

# Evaluate the model
mse_efa = mean_squared_error(y_test_efa, y_pred_efa)
r2_efa = r2_score(y_test_efa, y_pred_efa)

print(f"EFA Mean Squared Error: {mse_efa}")
print(f"EFA R-squared: {r2_efa}")

# Save the EFA regression results to a CSV file
results = pd.DataFrame({
    'Model': ['Linear Regression with EFA'],
    'Mean Squared Error': [mse_efa],
    'R-squared': [r2_efa]
})

# Save the results to the results folder
results_file_path = os.path.join(results_folder, 'efa_regression_model_performance.csv')
results.to_csv(results_file_path, index=False)

print(f"EFA regression results saved to {results_file_path}")
