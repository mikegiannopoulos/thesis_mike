"""
Bird Population Factor Analysis and Regression Script

This script performs factor analysis and linear regression to analyze the relationships 
between bird population data and various climate variables. It processes data for multiple 
species, applies exploratory factor analysis (EFA) to reduce dimensionality, and evaluates 
predictive models for each species.

Key functionalities:
1. Data Loading and Preprocessing:
   - Loads bird population and climate data from a CSV file.
   - Filters data by species and handles missing values.
   - Splits data into training and testing sets for robust evaluation.

2. Exploratory Factor Analysis (EFA):
   - Identifies key factors underlying climate variables using the Kaiser criterion.
   - Standardizes data and applies factor loadings to train and test datasets.

3. Regression Model Training and Evaluation:
   - Fits a linear regression model to factor-transformed data.
   - Computes metrics including Mean Squared Error (MSE) and R-squared (R²) for evaluation.

4. Visualization and Reporting:
   - Generates a unified heatmap of factor loadings across species and variables.
   - Creates radial profiles to illustrate loading patterns per species.
   - Produces small multiples of bar plots for cross-species factor loading comparisons.

Outputs:
- A results CSV containing factor loadings, regression metrics, and climate variable contributions.
- Visualizations saved as PNG files in the results directory.

Usage:
- Update the file_path variable with the path to the dataset.
- Ensure that the required columns (e.g., 'species', 'total_population', climate variables) 
  are present in the dataset.
- Specify the results_dir variable to define the directory for saving outputs.
"""


import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load and preprocess the dataset."""
    data = pd.read_csv(file_path)
    data = data.replace(-1, np.nan)  # Handle missing values
    return data

def preprocess_data(data, species_name, test_size=0.2):
    """Preprocess data with proper train-test split and imputation"""
    # Filter by species
    species_data = data[data['species'] == species_name].copy()
    if species_data.empty:
        return None, None, None, None
    
    # Split first to prevent data leakage
    X = species_data[['air_pressure_values', 'air_temperature_values', 'wind_values',
                      'sea_temp_values', 'seawater_level_values', 'wave_height_values']]
    y = species_data['total_population']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Impute missing values using training data statistics
    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)
    
    return X_train, X_test, y_train, y_test

def perform_efa(X_train, n_factors=None):
    """Perform EFA with dynamic factor selection"""
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Determine number of factors using Kaiser criterion
    if n_factors is None:
        fa = FactorAnalyzer(rotation=None, impute='drop')  # FIX 1: Add impute parameter
        fa.fit(X_scaled)
        ev, _ = fa.get_eigenvalues()
        n_factors = sum(ev > 1)  # Kaiser criterion

    # Perform factor analysis with explicit imputation
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', impute='drop')  # FIX 1
    fa.fit(X_scaled)
    
    return fa, scaler

def train_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'coefs': model.coef_
    }


def plot_unified_heatmap(results_df, results_dir):
    """Create a cleaner consolidated heatmap with hierarchical labels"""
    # Process data
    melt_df = results_df.melt(
        id_vars=['species', 'variable'],
        value_vars=[c for c in results_df.columns if c.startswith('factor_')],
        var_name='factor',
        value_name='loading'
    )
    
    # Create multi-level columns
    pivot_df = melt_df.pivot_table(
        index='species',
        columns=['variable', 'factor'],
        values='loading'
    )
    
    # Sort columns by variable then factor
    pivot_df = pivot_df.sort_index(axis=1, level=[0,1])
    
    # Create plot
    plt.figure(figsize=(18, len(pivot_df)*0.7))
    sns.heatmap(
        pivot_df,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'label': 'Loading Strength', 'shrink': 0.6},
        annot_kws={'size': 8}
    )
    
    # Formatting
    plt.title("Cross-Species Factor Loadings\n", fontsize=14, pad=20)
    plt.xlabel("Climate Variables → Factors", fontsize=12)
    plt.ylabel("Species", fontsize=12)
    
    # Rotate and align labels
    plt.xticks(
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        fontsize=10
    )
    
    # Add variable grouping labels
    ax = plt.gca()
    for idx, var in enumerate(pivot_df.columns.get_level_values(0).unique()):
        ax.annotate(
            var,
            xy=(idx * 3 + 1.5, -0.3),
            xycoords='data',
            annotation_clip=False,
            ha='center',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'simplified_heatmap.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Simplified heatmap saved to {plot_path}")

def plot_radial_profiles(results_df, results_dir, n_cols=3):
    """
    Create radial plots showing factor loading patterns for all species
    """
    # Get unique species and variables
    species_list = results_df['species'].unique()
    variables = results_df['variable'].unique()
    factors = [c for c in results_df.columns if c.startswith('factor_')]
    
    # Calculate grid dimensions
    n_rows = int(np.ceil(len(species_list) / n_cols))
    
    # Create figure
    fig = plt.figure(figsize=(n_cols*5, n_rows*5))
    fig.suptitle("Radial Factor Loading Profiles\n", fontsize=24, y=1.05)
    
    # Create axes grid
    for idx, species in enumerate(species_list, 1):
        ax = fig.add_subplot(n_rows, n_cols, idx, polar=True)
        
        # Get species data
        species_data = results_df[results_df['species'] == species]
        
        # Prepare angles (variables spaced equally around circle)
        angles = np.linspace(0, 2*np.pi, len(variables), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each factor as a separate line
        for factor in factors:
            values = species_data[factor].values.tolist()
            values += values[:1]  # Close the polygon
            ax.plot(angles, values, linewidth=2, linestyle='solid', 
                    label=factor.replace('_', ' ').title())
        
        # Format plot
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(variables, fontsize=12)
        ax.set_title(species, fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'radial_profiles.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Radial profiles saved to {plot_path}")

def plot_small_multiples(results_df, results_dir, n_cols=3):
    """
    Create a grid of bar plots showing factor loadings for all variables and species
    """
    # Melt data to long format
    melt_df = results_df.melt(
        id_vars=['species', 'variable'],
        value_vars=[c for c in results_df.columns if c.startswith('factor_')],
        var_name='factor',
        value_name='loading'
    ).dropna()
    
    # Clean factor names and ensure ordering
    melt_df['factor'] = melt_df['factor'].str.replace('_', ' ').str.title()
    variables_order = results_df['variable'].unique()
    
    # Create plot grid
    plt.figure(figsize=(18, len(melt_df['species'].unique())*1.5))
    g = sns.FacetGrid(
        melt_df,
        col='species',
        hue='factor',
        col_wrap=n_cols,
        palette='tab10',
        height=4,
        aspect=1.5,
        sharex=True,
        sharey=True
    )
    
    # Add bars
    g.map(sns.barplot, 'variable', 'loading', order=variables_order, ci=None)
    
    # Formatting
    g.set_xticklabels(rotation=45, ha='right', fontsize=14)
    g.set_axis_labels("", "Loading Strength")
    g.set_titles("{col_name}", fontsize=18, pad=10)
    g.add_legend(title='Factor', bbox_to_anchor=(1.1, 0.5))
    
    # Add global title
    plt.subplots_adjust(top=0.92)
    g.fig.suptitle('Factor Loadings Analysis by Species', fontsize=14)
    
    # Save
    plot_path = os.path.join(results_dir, 'small_multiples.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Small multiples plot saved to {plot_path}")


def main():
    file_path = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/data_for_correlation.csv'
    results_dir = '/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/bird_population_factor_analysis_regression/'
    os.makedirs(results_dir, exist_ok=True)
    
    data = load_data(file_path)
    results = []
    min_samples = 100  # Minimum required samples per species
    
    climate_vars = ['air_pressure_values', 'air_temperature_values', 'wind_values',
                    'sea_temp_values', 'seawater_level_values', 'wave_height_values']

    for species in data['species'].unique():
        print(f"\nProcessing {species}")
        
        # Get preprocessed data
        X_train, X_test, y_train, y_test = preprocess_data(data, species)
        
        # Check data adequacy - FIXED CONDITION
        if X_train is None or (isinstance(X_train, pd.DataFrame) and 
                              (X_train.empty or len(X_train) < min_samples)):
            sample_count = len(X_train) if X_train is not None else 0
            print(f" Insufficient data ({sample_count} samples). Skipping.")
            continue
            
        try:
            # Perform EFA on training data
            fa, scaler = perform_efa(X_train)
            
            # Transform both train and test data
            X_train_factors = fa.transform(scaler.transform(X_train))
            X_test_factors = fa.transform(scaler.transform(X_test))
            
            # Check for valid transformed data
            if X_train_factors.shape[0] == 0 or X_test_factors.shape[0] == 0:
                print(" No valid data after factor analysis. Skipping.")
                continue
                
            # Train and evaluate model
            model_results = train_evaluate_model(X_train_factors, X_test_factors, y_train, y_test)
            
            # Store results - FIXED COLUMN NAMES
            loadings = fa.loadings_
            for var_idx, var_name in enumerate(climate_vars):
                result_entry = {
                    'species': species,
                    'variable': var_name,
                    'mse': model_results['mse'],
                    'r2': model_results['r2']
                }
                # Add factors dynamically
                for i in range(loadings.shape[1]):
                    result_entry[f'factor_{i+1}'] = loadings[var_idx, i]
                results.append(result_entry)
            
                       
        except Exception as e:
            print(f" Error processing {species}: {str(e)}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
    
    # Generate visualizations
    plot_unified_heatmap(results_df, results_dir)  
    plot_radial_profiles(results_df, results_dir, n_cols=3)
    plot_small_multiples(results_df, results_dir, n_cols=3)
    
    print("\nAnalysis complete. Results saved to:", results_dir)

if __name__ == "__main__":
    main()