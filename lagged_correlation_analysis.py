"""
lagged_correlation_analysis.py

This script performs a lagged correlation analysis between bird population data and climate variables.
It creates lagged versions of each climate variable (e.g., 14, 28, 42 days), computes species-specific
correlations, and then produces a heatmap of all lagged correlations. Additionally, it generates a grid
of line plots showing how the correlation for each species (with multiple climate variables) evolves 
across a range of lag values.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats  
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the CSV data and creates a 'survey_date' column assuming surveys occur on June 1 of the given Year.
    """
    df = pd.read_csv(file_path)
    df['survey_date'] = pd.to_datetime(df['Year'].astype(str) + '-06-01')
    return df

def create_lagged_variables(df: pd.DataFrame, climate_vars: list, lag_days: list) -> pd.DataFrame:
    """
    Creates lagged versions of the specified climate variables for the given lag days.
    """
    df_lagged = df.copy()
    for lag in lag_days:
        for var in climate_vars:
            df_lagged[f"{var}_lag{lag}"] = df_lagged[var].shift(lag)
    return df_lagged

def compute_species_correlations(
    df: pd.DataFrame,
    species_col: str,
    population_col: str,
    climate_vars: list,
    lag_days: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns tuple: (correlation_df, pvalue_df)
    """
    species_groups = df.groupby(species_col)
    corr_results = {}
    pvalue_results = {}
    
    for sp, group in species_groups:
        corr_dict = {}
        pvalue_dict = {}
        for var in climate_vars:
            for lag in lag_days:
                lagged_var = f"{var}_lag{lag}"
                valid_data = group[[population_col, lagged_var]].dropna()
                if len(valid_data) >= 3:  # Minimum for Pearson correlation
                    corr, pval = stats.pearsonr(valid_data[population_col], valid_data[lagged_var])
                else:
                    corr, pval = np.nan, np.nan
                corr_dict[lagged_var] = corr
                pvalue_dict[lagged_var] = pval
        corr_results[sp] = corr_dict
        pvalue_results[sp] = pvalue_dict
    
    return pd.DataFrame(corr_results).T, pd.DataFrame(pvalue_results).T

def extract_significant_leads(
    df_corr: pd.DataFrame, 
    df_pvalues: pd.DataFrame, 
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Identifies ALL statistically significant correlations for each species.
    Returns DataFrame with species, variable, correlation, p-value, and direction.
    """
    results = []
    
    for species in df_corr.index:
        # Get all correlations and p-values for this species
        corrs = df_corr.loc[species]
        pvals = df_pvalues.loc[species]
        
        # Find all significant correlations
        significant_mask = (pvals < alpha) & (~corrs.isna())
        significant_vars = corrs[significant_mask].index.tolist()
        
        if not significant_vars:
            # No significant correlations found
            results.append({
                'species': species,
                'variable': None,
                'correlation': None,
                'p_value': None,
                'direction': None,
                'significance': 'non-significant'
            })
            continue
            
        # Add all significant correlations
        for var in significant_vars:
            corr_value = corrs[var]
            p_value = pvals[var]
            results.append({
                'species': species,
                'variable': var,
                'correlation': corr_value,
                'p_value': p_value,
                'direction': 'positive' if corr_value > 0 else 'negative',
                'significance': 'significant'
            })
    
    df_sig = pd.DataFrame(results)
    
    # Sort by species and correlation strength
    return df_sig.sort_values(by=['species', 'correlation'], 
                            key=lambda x: x.abs() if x.name == 'correlation' else x,
                            ascending=[True, False])

def plot_heatmap(df_corr: pd.DataFrame, title: str = "Lagged Correlation Heatmap") -> None:
    """
    Plots a heatmap for the given correlation DataFrame.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='RdBu', center=0, vmin=-1, vmax=1)
    plt.title(title)
    plt.ylabel("Species")
    plt.xlabel("Lagged Climate Variables")
    plt.tight_layout()
    plt.show()

def compute_lagged_correlation_for_species(df: pd.DataFrame, species: str, climate_var: str, lag: int, species_col: str, population_col: str) -> float:
    """
    Computes the Pearson correlation for a given species, climate variable, and lag.
    """
    df_species = df[df[species_col] == species].copy()
    lagged_col = f"{climate_var}_lag_temp"
    df_species[lagged_col] = df_species[climate_var].shift(lag)
    return df_species[population_col].corr(df_species[lagged_col], method='pearson')

def plot_lagged_correlation_grid(
    df: pd.DataFrame,
    species_list: list,
    climate_vars: list,
    lag_range: list,
    species_col: str,
    population_col: str
) -> pd.DataFrame:
    """
    Plots a grid of line plots and returns a DataFrame with correlation and p-value data.
    """
    data = []
    n_species = len(species_list)
    n_cols = 3
    n_rows = int(np.ceil(n_species / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, species in enumerate(species_list):
        ax = axes[i]
        for var in climate_vars:
            correlations = []
            p_values = []
            for lag in lag_range:
                df_species = df[df[species_col] == species].copy()
                lagged_col = f"{var}_lag_temp"
                df_species[lagged_col] = df_species[var].shift(lag)
                
                # Calculate both correlation and p-value
                valid_data = df_species[[population_col, lagged_col]].dropna()
                if len(valid_data) >= 3:
                    corr, pval = stats.pearsonr(valid_data[population_col], valid_data[lagged_col])
                else:
                    corr, pval = (np.nan, np.nan)
                
                correlations.append(corr)
                p_values.append(pval)
                data.append({
                    "species": species,
                    "climate_variable": var,
                    "lag": lag,
                    "correlation": corr,
                    "p_value": pval
                })
            
            ax.plot(lag_range, correlations, marker='.', label=var)
        
        ax.set_title(species)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Correlation")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend(fontsize=8)
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig("/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/lagged_correlation_analysis/lagged_correlation_grid.png", 
                dpi=300, bbox_inches="tight")  
    plt.show()
    
    return pd.DataFrame(data)

def main():
    # Define parameters and file paths
    output_dir = "/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/lagged_correlation_analysis/"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = "/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/data_for_correlation.csv"
    species_col = "species"
    population_col = "total_population"
    climate_vars = [
        'air_pressure_values', 
        'air_temperature_values', 
        'wind_values', 
        'sea_temp_values', 
        'seawater_level_values', 
        'wave_height_values'
    ]
    lag_days = [30, 60, 90]
    lag_range = list(range(0, 90, 7))
    
    # Load and prepare data
    df = load_data(file_path)
    df_lagged = create_lagged_variables(df, climate_vars, lag_days)
    df_lagged_clean = df_lagged.dropna()
    
    # Compute correlations and p-values for heatmap
    df_correlations, df_pvalues = compute_species_correlations(
        df_lagged_clean,
        species_col,
        population_col,
        climate_vars,
        lag_days
    )
    
    # Save heatmap data
    df_correlations.to_csv(f"{output_dir}heatmap_correlations.csv")
    df_pvalues.to_csv(f"{output_dir}heatmap_pvalues.csv")
    
    # Compute and save best leads (using improved version)
    df_significant = extract_significant_leads(df_correlations, df_pvalues)
    df_significant.to_csv(f"{output_dir}significant_leads.csv", index=False)
    
    # Generate and save line plot data with p-values
    species_list = df_lagged_clean[species_col].unique().tolist()
    line_data = plot_lagged_correlation_grid(
        df_lagged_clean,  # Use cleaned data
        species_list,
        climate_vars,
        lag_range,
        species_col,
        population_col
    )
    line_data.to_csv(f"{output_dir}line_correlations_with_pvalues.csv", index=False)

if __name__ == "__main__":
    main()
