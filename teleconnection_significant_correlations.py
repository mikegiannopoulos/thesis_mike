"""
Bird Population vs. Teleconnection Patterns Analysis 

This script analyzes the correlation between bird population data and 
four Teleconnection Pattern indexes on a monthly lag basis. 

### Workflow:
1. **Load & Parse Data**:
   - Reads species correlation results from a CSV file.
   - Transforms the data into a long format for easier processing.
   - Extracts key components from metric column names (Index, Correlation Type, Lag).

2. **Multiple Testing Correction**:
   - Performs Bonferroni and Benjamini-Hochberg (FDR) corrections to adjust p-values.
   - Identifies significant correlations using raw p-values and corrected thresholds.

3. **Seasonal Mapping**:
   - Maps correlation lags to corresponding months and seasons.

4. **Generate Insights**:
   - Identifies significant correlations based on raw p-values.
   - Analyzes the distribution of significant correlations across seasons and months.
   - Computes species-specific seasonal sensitivity.

5. **Save & Visualize**:
   - Saves significant correlation results as a CSV file.
   - Generates and saves a bar plot of seasonal correlation counts per index.
   - Creates and saves a heatmap showing species-specific seasonal sensitivity.

### Outputs:
- `significant_correlations.csv`: Contains all significant correlation results.
- `seasonal_counts.png`: Bar plot showing significant correlations by season.
- `species_season_heatmap.png`: Heatmap of species-specific seasonal sensitivity.

### Configuration:
- The output directory for results can be set using the `output_dir` variable.
"""


import pandas as pd
import numpy as np
import os
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------
# 0. Configuration
# ------------------
output_dir = "/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/bird_population_vs_nao_correlation_monthly/data_handling"  # <-- Specify output folder here
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist


# ------------------
# 1. Load & Parse Data
# ------------------
df = pd.read_csv('/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/bird_population_vs_nao_correlation_monthly/species_correlation_results.csv')

# Melt to long format for easier processing
id_vars = ['species']
value_vars = [col for col in df.columns if col != 'species']
melted = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='metric', value_name='value')

# Split metric into components: Index_CorrelationType_P_LagXM or Index_CorrelationType_LagXM
melted[['Index', 'CorrelationType', 'Lag']] = melted['metric'].str.extract(
    r'^([A-Z]+)_(Pearson|Spearman)(_P)?_Lag(\d+)M$'
)[[0, 1, 3]].rename(columns={0: 'Index', 1: 'CorrelationType', 3: 'Lag'})
melted['Lag'] = melted['Lag'].astype(int)
melted['Metric'] = melted['metric'].str.contains('_P_').map({True: 'p_value', False: 'coefficient'})

# Pivot to separate coefficients and p-values
pivoted = melted.pivot_table(
    index=['species', 'Index', 'CorrelationType', 'Lag'],
    columns='Metric',
    values='value',
    aggfunc='first'
).reset_index()

# ------------------
# 2. Multiple Testing Correction
# ------------------
# Total number of tests
n_tests = pivoted.shape[0]
bonferroni_threshold = 0.05 / n_tests

# Apply Benjamini-Hochberg FDR
p_values = pivoted['p_value'].values
_, fdr_adjusted_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
pivoted['p_value_fdr'] = fdr_adjusted_p
pivoted['significant_raw'] = pivoted['p_value'] < 0.05
pivoted['significant_bonferroni'] = pivoted['p_value'] < bonferroni_threshold
pivoted['significant_fdr'] = pivoted['p_value_fdr'] < 0.05

# ------------------
# 3. Seasonal Mapping
# ------------------
def lag_to_season(lag):
    if lag in [1, 2, 12]:
        return 'Winter'
    elif lag in [3, 4, 5]:
        return 'Spring'
    elif lag in [6, 7, 8]:
        return 'Summer'
    elif lag in [9, 10, 11]:
        return 'Autumn'

pivoted['Month'] = pivoted['Lag'].apply(lambda x: pd.to_datetime(f'2023-{x}-1').month_name())
pivoted['Season'] = pivoted['Lag'].apply(lag_to_season)

# ------------------
# 4. Generate Insights
# ------------------
# Filter significant results (using raw p-value < 0.05)
significant = pivoted[pivoted['significant_raw']].copy()

# Seasonal analysis
seasonal_counts = significant.groupby(['Index', 'Season']).size().reset_index(name='counts')
monthly_counts = significant.groupby(['Index', 'Month']).size().reset_index(name='counts')

# Species-season sensitivity
species_season = significant.groupby(['species', 'Season']).size().reset_index(name='counts')
heatmap_data = species_season.pivot(index='species', columns='Season', values='counts').fillna(0)

# ------------------
# 5. Save & Visualize
# ------------------
# Save all significant results
significant_path = os.path.join(output_dir, "significant_correlations.csv")
significant.to_csv(significant_path, index=False)

# Plot 1: Seasonal Counts by Index
plt.figure(figsize=(10, 6))
sns.barplot(data=seasonal_counts, x='Index', y='counts', hue='Season')
plt.title('Significant Correlations by Season and Index')
seasonal_plot_path = os.path.join(output_dir, "seasonal_counts.png")
plt.savefig(seasonal_plot_path)
plt.close()

# Plot 2: Species Sensitivity Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
plt.title('Species-Specific Seasonal Sensitivity')
heatmap_path = os.path.join(output_dir, "species_season_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

print(f"Analysis complete! Results saved to: {os.path.abspath(output_dir)}")