"""
Perform regression analysis using Random Forest and Gradient Boosting for each species.
Evaluate models and save results, including visualizations of MSE and R² values.

Guidelines followed:
1. Analyze each species individually.
2. Evaluate models with MSE and R² metrics and save results in a CSV file.
3. Create visualizations comparing model performance across species.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Constants
FEATURE_COLUMNS = [
    'air_pressure_values', 'air_temperature_values', 'wind_values',
    'sea_temp_values', 'seawater_level_values', 'wave_height_values'
]
TARGET_COLUMN = 'total_population'
MODELS = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}
RESULTS_FOLDER = '/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/bird_population_random_forest_and_gradient_boosting'

class DataManager:
    """Handles data loading and grouping by species"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.grouped_data = None
        self.global_means = {}

    def load_and_group(self):
        self.data = pd.read_csv(self.file_path).replace(-1, np.nan)
        self._calculate_global_means()
        self.grouped_data = self.data.groupby('species')

    def _calculate_global_means(self):
        for col in FEATURE_COLUMNS:
            self.global_means[col] = self.data[col].mean()

    def get_species_groups(self):
        return self.grouped_data.groups.keys()

    def get_species_data(self, species_name):
        return self.grouped_data.get_group(species_name).copy()

import logging

class DataPreprocessor:
    """Handles data preprocessing for individual species."""
    def __init__(self, global_means):
        self.global_means = global_means
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def preprocess(self, species_data):
        try:
            processed_data = species_data.copy()
            for col in FEATURE_COLUMNS:
                species_mean = processed_data[col].mean()
                fill_value = species_mean if not pd.isna(species_mean) else self.global_means[col]
                processed_data[col].fillna(fill_value, inplace=True)
            logging.info(f"Preprocessed data for species: {species_data['species'].iloc[0]}")
            return processed_data[FEATURE_COLUMNS], processed_data[TARGET_COLUMN]
        except Exception as e:
            logging.error(f"Error preprocessing data for species: {e}")
            return pd.DataFrame(), pd.Series()


class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self, models):
        self.models = models
        self.scaler = StandardScaler()

    def train_and_evaluate(self, X, y):
        if len(X) < 2:
            return []
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        results = []
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({
                'model': model_name,
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            })
        return results

class ResultHandler:
    """Handles result storage and visualization"""
    def __init__(self, results_folder):
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def save_results(self, results):
        df = pd.DataFrame(results)
        file_path = os.path.join(self.results_folder, 'model_performance.csv')
        df.to_csv(file_path, index=False)
        return file_path

    def visualize_results(self, results):
        df = pd.DataFrame(results)
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x='species', y='mse', hue='model', data=df)
        plt.title('MSE Comparison')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.barplot(x='species', y='r2', hue='model', data=df)
        plt.title('R² Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_folder, 'performance_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path



def main():
    # Initialize components
    data_mgr = DataManager('/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/data_for_correlation.csv')
    data_mgr.load_and_group()
    
    result_handler = ResultHandler(RESULTS_FOLDER)
    model_trainer = ModelTrainer(MODELS)
    preprocessor = DataPreprocessor(data_mgr.global_means)
    
    all_results = []
    
    for species in data_mgr.get_species_groups():
        species_data = data_mgr.get_species_data(species)
        X, y = preprocessor.preprocess(species_data)
        
        if len(X) == 0 or len(y) == 0:
            continue
            
        metrics = model_trainer.train_and_evaluate(X, y)
        for metric in metrics:
            metric['species'] = species
            all_results.append(metric)
    
    # Save and visualize results
    csv_path = result_handler.save_results(all_results)
    plot_path = result_handler.visualize_results(all_results)
    
    print(f"Results saved to: {csv_path}")
    print(f"Visualization saved to: {plot_path}")

if __name__ == "__main__":
    main()