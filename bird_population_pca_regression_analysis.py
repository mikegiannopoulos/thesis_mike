import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
final_bird_data = pd.read_csv('/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/data_for_correlation.csv')

# Replace -1 with NaN and group by species
final_bird_data = final_bird_data.replace(-1, np.nan)
species_groups = final_bird_data.groupby('species')

# Define minimum samples required per species to include in modeling
MIN_SAMPLES_PER_SPECIES = 50  # Adjust based on your data
results = []

for species_name, group in species_groups:
    # Skip species with too few samples
    if len(group) < MIN_SAMPLES_PER_SPECIES:
        print(f"Skipping {species_name} (only {len(group)} samples).")
        continue
    
    # Select features and target
    X = group[['air_pressure_values', 'air_temperature_values', 
               'wind_values', 'sea_temp_values', 'seawater_level_values', 
               'wave_height_values']]
    y = group['total_population']
    
    # Split into train/test (stratify if needed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Impute missing values using training data means
    train_means = X_train.mean()
    X_train_imputed = X_train.fillna(train_means)
    X_test_imputed = X_test.fillna(train_means)  # Use training means to avoid leakage
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Apply PCA (retain 90% variance)
    pca = PCA(n_components=0.90)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_pca, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_pca)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results.append({
        'Species': species_name,
        'Mean Squared Error': mse,
        'R-squared': r2,
        'Number of Samples': len(group)
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_folder = '/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/bird_population_pca_regression_analysis'
results_df.to_csv(os.path.join(results_folder, 'pca_regression_per_species.csv'), index=False)

# Print summary
print("\n=== Results Summary ===")
print(f"Total species analyzed: {len(results_df)}")
print(f"Average MSE: {results_df['Mean Squared Error'].mean():.2f}")
print(f"Average R²: {results_df['R-squared'].mean():.4f}")