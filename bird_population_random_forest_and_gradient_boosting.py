"""
This script is used to perform regression analysis on the bird data using two different models: 
Random Forest Regressor and Gradient Boosting Regressor. 

The data is split into training and testing sets and the performance of each model is evaluated using 
the mean squared error (MSE) and the R-squared value.
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

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

# Select relevant columns for regression
X = filtered_data[['air_pressure_values', 'air_temperature_values', 
                   'wind_values', 'sea_temp_values', 'seawater_level_values', 
                   'wave_height_values']]
y = filtered_data['individualCount']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create the results folder path
results_folder = '/home/michael/Education/UoG/Earth Science Master/Thesis/results/'

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 1. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R-squared: {r2_rf}")

# 2. Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Evaluate Gradient Boosting
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f"Gradient Boosting - Mean Squared Error: {mse_gb}")
print(f"Gradient Boosting - R-squared: {r2_gb}")

# Save the results to a CSV file
results = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting'],
    'Mean Squared Error': [mse_rf, mse_gb],
    'R-squared': [r2_rf, r2_gb]
})

# Save the results to the results folder
results_file_path = os.path.join(results_folder, 'regression_model_performance.csv')
results.to_csv(results_file_path, index=False)

print(f"Results saved to {results_file_path}")
