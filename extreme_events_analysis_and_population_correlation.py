"""
This script processes climate data from SMHI and correlates the extreme events
with bird population data from the bird database.

The script loads the climate data, identifies extreme events, and summarizes
the extreme events per year. It then correlates each variable with the bird
population and saves the correlation matrix. The script also visualizes the
correlation matrix and the time series of extreme events.

The results are saved to a folder specified by the user.
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Single Responsibility: Load data from file
def load_data(data_path):
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Single Responsibility: Identify extreme events in the dataset
def identify_extreme_events(data, threshold=0.95):
    data_long = pd.melt(data, id_vars=['Date'], var_name='Station', value_name='Value')
    extreme_threshold = data_long['Value'].quantile(threshold)
    data_long['Extreme_Event'] = data_long['Value'] > extreme_threshold
    daily_extreme_events = data_long.groupby('Date')['Extreme_Event'].mean().reset_index()
    return daily_extreme_events

# Single Responsibility: Summarize extreme events by year
def summarize_extreme_events(data_long, variable_name):
    summary = data_long.groupby(data_long['Date'].dt.year)['Extreme_Event'].sum().reset_index()
    summary.columns = ['Year', variable_name]
    return summary

# Single Responsibility: Correlate extreme events with bird population data for each species
def correlate_with_population_by_species(extreme_summary, population_data):
    extreme_summary = extreme_summary.rename(columns={extreme_summary.columns[1]: 'Extreme_Events'})
    results = []
    
    # Group population data by species and year
    species_population = population_data.groupby(['species', 'Year'])['total_population'].sum().reset_index()
    
    # Iterate over each species and calculate correlation
    for species in species_population['species'].unique():
        species_data = species_population[species_population['species'] == species]
        merged_data = pd.merge(extreme_summary, species_data, on='Year')
        
        # Calculate correlation coefficient and p-value
        corr_coef, p_value = pearsonr(merged_data['Extreme_Events'], merged_data['total_population'])
        results.append({'Species': species, 'Correlation Coefficient': corr_coef, 'P-value': p_value})
    
    # Convert results to DataFrame
    correlation_results = pd.DataFrame(results)
    return correlation_results

# Single Responsibility: Plot and save correlation matrix
def plot_correlation_matrix(correlation_matrix, results_folder):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Between Climate Variables and Bird Population", fontsize=16)
    plt.tight_layout()

    # Save to results folder
    plt.savefig(os.path.join(results_folder, "climate_variable_population_correlation_heatmap.png"), dpi=300)
    plt.show()

# Single Responsibility: Plot time series of extreme events
def plot_time_series(data, results_folder):
    data.set_index('Year', inplace=True)
    data.plot(figsize=(12, 8), marker='o')
    plt.title("Extreme Events Over Time for Climate Variables", fontsize=16)
    plt.ylabel("Extreme Event Count")
    plt.legend(title="Climate Variables", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save to results folder
    plt.savefig(os.path.join(results_folder, "extreme_events_time_series.png"), dpi=300)
    plt.show()


# Single Responsibility: Plot grid of heatmaps for species-specific correlations
def plot_species_correlation_heatmaps(species_correlation_results, results_folder):
    import math

    # Get unique species
    species_list = species_correlation_results['Species'].unique()
    num_species = len(species_list)
    
    # Determine grid size
    cols = 3  # Number of columns in the grid
    rows = math.ceil(num_species / cols)
    
    # Create subplots and adjust layout
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
    fig.suptitle("Comparative Correlations Across Species and Climate Variables", 
             fontsize=24, fontweight='bold', y=0.95)  # Adjust 'y' to control vertical position
    fig.subplots_adjust(wspace=0.15, hspace=0.15, top=0.9, bottom=0.1, left=0.05, right=0.85)  # Adjust layout margins
    
    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Create a shared colorbar axis
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Adjust position and size of the colorbar
    
    for i, species in enumerate(species_list):
        ax = axes[i]
        
        # Filter data for the species
        species_data = species_correlation_results[species_correlation_results['Species'] == species]
        
        # Pivot the data for the heatmap
        heatmap_data = species_data.pivot(index='Variable', columns='Species', values='Correlation Coefficient')
        
        # Plot heatmap
        sns.heatmap(
            heatmap_data, annot=True, cmap='viridis', fmt=".2f", ax=ax, cbar=(i == 0),
            vmin=-0.8, vmax=0.8, cbar_ax=(cbar_ax if i == 0 else None),  # Shared colorbar
            annot_kws={"size": 16}  # Set font size for annotations
        )

        ax.set_title(f"{species}", fontsize=18)
        
        # Set Y-axis tick labels only for the first heatmap in each row
        if i % cols == 0:  # First heatmap in the row
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Keep labels horizontal
        else:
            ax.set_yticklabels([])  # Remove Y-axis labels for other heatmaps in the row

        # Remove X-axis label and ticks
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticks([])  # Remove X-axis ticks
    
    # Remove any unused axes
    for j in range(num_species, len(axes)):
        axes[j].remove()  # Completely remove unused subplot axes
    
    # Save the grid of heatmaps
    output_path = os.path.join(results_folder, 'species_correlation_heatmaps_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Use bbox_inches='tight' for proper layout
    plt.show()



# Update process_climate_data to handle species-specific correlations
def process_climate_data(files, base_data_path, population, output_folder):
    combined_extreme_events = pd.DataFrame()
    species_correlation_results = pd.DataFrame()
    
    for file_name, variable_name in files.items():
        data_path = os.path.join(base_data_path, file_name)
        data = load_data(data_path)
        extreme_events = identify_extreme_events(data)
        extreme_summary = summarize_extreme_events(extreme_events, variable_name)
        combined_extreme_events = merge_extreme_event_data(combined_extreme_events, extreme_summary)
        
        # Get species-specific correlations
        correlation_results = correlate_with_population_by_species(extreme_summary, population)
        correlation_results['Variable'] = variable_name  # Add variable name for context
        species_correlation_results = pd.concat([species_correlation_results, correlation_results], ignore_index=True)
    
    return combined_extreme_events, species_correlation_results

# Single Responsibility: Merge extreme events summaries
def merge_extreme_event_data(combined_extreme_events, extreme_summary):
    if combined_extreme_events.empty:
        combined_extreme_events = extreme_summary
    else:
        combined_extreme_events = pd.merge(combined_extreme_events, extreme_summary, on='Year', how='outer')
    return combined_extreme_events

# Single Responsibility: Save results to the results folder
def save_results(combined_extreme_events, correlation_matrix, species_correlation_results, results_folder):
    # Save combined extreme event summary and correlation matrix to the results folder
    combined_extreme_events.to_csv(os.path.join(results_folder, 'combined_extreme_event_summary.csv'), index=False)
    correlation_matrix.to_csv(os.path.join(results_folder, 'climate_variable_population_correlation_matrix.csv'))
    
    # Save species-specific correlation results
    species_correlation_results.to_csv(os.path.join(results_folder, 'species_correlation_results.csv'), index=False)

def main():
    files = {
        'air_temperature.csv': 'Air Temperature',
        'air_pressure.csv': 'Air Pressure',
        'sea_temp.csv': 'Sea Temperature',
        'wave_height.csv': 'Wave Height',
        'seawater_level.csv': 'Seawater Level',
        'wind.csv': 'Wind'
    }
    
    base_data_path = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/SMHI'
    population_path = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/data_for_correlation.csv'
    
    # Keep original column names
    population = pd.read_csv(population_path)
    
    results_folder = '/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results'
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    combined_extreme_events, species_correlation_results = process_climate_data(files, base_data_path, population, results_folder)
    
    population_summary = population.groupby('Year')['total_population'].sum().reset_index()
    combined_extreme_events = pd.merge(combined_extreme_events, population_summary, on='Year')
    
    correlation_matrix = combined_extreme_events.drop(columns=['Year']).corr()
    print("\nCorrelation matrix between climate variables and bird population:")
    print(correlation_matrix)
    
    # Save results and visualizations
    save_results(combined_extreme_events, correlation_matrix, species_correlation_results, results_folder)
    plot_correlation_matrix(correlation_matrix, results_folder)
    plot_time_series(combined_extreme_events, results_folder)
    plot_species_correlation_heatmaps(species_correlation_results, results_folder)

    print(f"All results and visualizations saved to '{results_folder}'")

if __name__ == "__main__":
    main()
