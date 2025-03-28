import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Import the necessary SciPy statistics functions
from scipy.stats import pearsonr, spearmanr, kendalltau

def grouped_heatmaps(data_file_path, save_dir='/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/correlation_heatmaps', save_images=False):
    print(f"Reading data from {data_file_path}...")
    cor_data = pd.read_csv(data_file_path)
    
    # Replace -1 with NaN
    filtered_data = cor_data.replace(-1, np.nan)
    
    # Check for the "species" column
    if "species" not in filtered_data.columns:
        raise ValueError("The dataset must include a 'species' column.")
    
    # Abbreviated column names
    abbreviations = {
        'total_population': 'population',
        'air_pressure_values': 'air press',
        'air_temperature_values': 'air temp',
        'wind_values': 'wind',
        'sea_temp_values': 'sea temp',
        'seawater_level_values': 'sea level',
        'wave_height_values': 'wave height'
    }
    
    # Rename columns
    filtered_data.rename(columns=abbreviations, inplace=True)
    
    # Group data by species
    grouped = filtered_data.groupby('species')

    # Initialize list to collect all correlation data for CSV
    all_correlations = []

    # -------------------------------------------------------------------------
    # Helper function to compute correlation matrix AND p-value matrix
    # -------------------------------------------------------------------------
    def correlation_with_pvals(df, method='pearson'):
        """Compute the correlation and associated p-values for each pair of columns."""
        # Get numeric columns only (if your data has non-numeric columns beyond 'species')
        cols = df.select_dtypes(include=[np.number]).columns

        corr_vals = np.zeros((len(cols), len(cols)))
        pvals = np.zeros((len(cols), len(cols)))

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                # To avoid re-computing for symmetric cells
                if j < i:
                    corr_vals[i,j] = corr_vals[j,i]
                    pvals[i,j] = pvals[j,i]
                elif j == i:
                    # Correlation with self
                    corr_vals[i,j] = 1.0
                    pvals[i,j] = 0.0
                else:
                    # Drop NaN in pairwise fashion
                    valid_data = df[[col1, col2]].dropna()
                    if len(valid_data) < 2:
                        # Not enough data to compute correlation
                        corr_vals[i,j] = np.nan
                        pvals[i,j] = np.nan
                        continue

                    x = valid_data[col1]
                    y = valid_data[col2]

                    # Compute correlation and p-value
                    if method == 'pearson':
                        r, p = pearsonr(x, y)
                    elif method == 'spearman':
                        r, p = spearmanr(x, y)
                    elif method == 'kendall':
                        r, p = kendalltau(x, y)
                    else:
                        raise ValueError("Unsupported correlation method. Choose from 'pearson', 'spearman', or 'kendall'.")

                    corr_vals[i,j] = r
                    corr_vals[j,i] = r
                    pvals[i,j] = p
                    pvals[j,i] = p

        # Create DataFrames from arrays
        corr_df = pd.DataFrame(corr_vals, index=cols, columns=cols)
        pval_df = pd.DataFrame(pvals, index=cols, columns=cols)

        return corr_df, pval_df
    # -------------------------------------------------------------------------

    def generate_combined_heatmaps(correlation_type, method_name):
        print(f"Generating {method_name} correlation heatmaps...")
        species_count = len(grouped)
        cols = 4  # Number of columns in the grid
        rows = (species_count // cols) + (1 if species_count % cols != 0 else 0)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()
        
        for idx, (species, group) in enumerate(grouped):
            if idx >= len(axes):
                break
            
            # Select relevant columns for correlation
            correlation_data = group[list(abbreviations.values())]
            
            # Skip species with insufficient data
            if correlation_data.isnull().all().any():
                print(f"Skipping {species} due to insufficient data.")
                # Turn subplot off
                axes[idx].axis('off')
                continue
            
            # Calculate correlation and p-value matrices
            corr_matrix, pval_matrix = correlation_with_pvals(correlation_data, method=correlation_type)

            # If you'd like to filter out columns that weren't numeric or ended up empty,
            # you can do so here (corr_matrix might be empty if no numeric columns).
            if corr_matrix.empty:
                print(f"Skipping {species} because correlation matrix is empty.")
                axes[idx].axis('off')
                continue

            # ---- 1) Melt correlation matrix to long format  ----
            corr_long = (
                corr_matrix
                .reset_index()
                .melt(
                    id_vars='index', 
                    var_name='variable2', 
                    value_name='correlation'
                )
                .rename(columns={'index': 'variable1'})
            )

            # ---- 2) Melt p-value matrix to long format  ----
            pval_long = (
                pval_matrix
                .reset_index()
                .melt(
                    id_vars='index', 
                    var_name='variable2', 
                    value_name='p_value'
                )
                .rename(columns={'index': 'variable1'})
            )

            # ---- 3) Merge correlation & p-value data  ----
            merged_long = pd.merge(
                corr_long, pval_long,
                on=['variable1', 'variable2'],
                how='left'
            )
            merged_long['species'] = species
            merged_long['method'] = correlation_type

            # Append to the global collector
            all_correlations.append(merged_long)
            
            # Plot heatmap of correlations
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=0.5, 
                ax=axes[idx],
                cbar=False
            )
            axes[idx].set_title(species, fontsize=12)
            
            # Axis labels adjustments
            if idx % cols == 0:
                axes[idx].set_yticklabels(corr_matrix.index, rotation=0)
            else:
                axes[idx].set_yticks([])
            
            if idx >= (rows - 1) * cols:
                axes[idx].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            else:
                axes[idx].set_xticks([])
        
        # Hide unused subplots
        for ax in axes[species_count:]:
            ax.axis('off')
        
        # Set overall title and adjust layout
        plt.suptitle(f'{method_name} Correlation Heatmaps', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save or show
        if save_images:
            save_path = os.path.join(save_dir, f'{correlation_type}_correlation_combined.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved {method_name} heatmaps as {save_path}")
        plt.show()

    # Generate combined heatmaps for each correlation type
    generate_combined_heatmaps('pearson', 'Pearson')
    generate_combined_heatmaps('spearman', 'Spearman')
    generate_combined_heatmaps('kendall', 'Kendall Tau')

    # Save all correlations to CSV
    if all_correlations:
        combined_df = pd.concat(all_correlations, ignore_index=True)
        
        # Save all correlations to CSV
        csv_path = os.path.join(save_dir, 'correlation_results_with_pvals.csv')
        combined_df.to_csv(csv_path, index=False)
        print(f"\nSaved comprehensive correlation results (with p-values) to {csv_path}")
    
        # ---------------------------------------------------
        # Add lines for creating the "significant results" CSV
        # ---------------------------------------------------
        significance_level = 0.05
        significant_df = combined_df[
            (combined_df['p_value'] < significance_level) &
            (combined_df['variable1'] != combined_df['variable2'])
        ]
    
        # Save only the significant results
        significant_csv_path = os.path.join(save_dir, 'correlation_results_significant.csv')
        significant_df.to_csv(significant_csv_path, index=False)
    
        print(f"\nSaved significant correlation results (p < {significance_level}) to {significant_csv_path}")
    
    else:
        print("\nNo correlation data available to save.")

if __name__ == '__main__':
    data_file_path = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/data_for_correlation.csv'
    save_dir = '/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/correlation_heatmaps'
    save_images = True

    print(f"Running the grouped heatmap generation script with the following options:\nData path: {data_file_path}\nSave directory: {save_dir}\nSave images: {save_images}")

    grouped_heatmaps(
        data_file_path=data_file_path,
        save_dir=save_dir,
        save_images=save_images
    )
