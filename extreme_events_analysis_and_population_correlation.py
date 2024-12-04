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

# Single Responsibility: Correlate extreme events with bird population data, including p-value
def correlate_with_population(extreme_summary, population_data):
    extreme_summary = extreme_summary.rename(columns={extreme_summary.columns[1]: 'Extreme_Events'})
    population_summary = population_data.groupby('Year')['Bird Population'].sum().reset_index()
    merged_data = pd.merge(extreme_summary, population_summary, on='Year')
    
    # Calculate correlation coefficient and p-value
    corr_coef, p_value = pearsonr(merged_data['Extreme_Events'], merged_data['Bird Population'])
    correlation = pd.DataFrame({
        'Correlation Coefficient': [corr_coef],
        'P-value': [p_value]
    })
    
    return correlation

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

# Open-Closed Principle: Processing and correlating climate data is abstracted
def process_climate_data(files, base_data_path, population, output_folder):
    combined_extreme_events = pd.DataFrame()
    for file_name, variable_name in files.items():
        data_path = os.path.join(base_data_path, file_name)
        data = load_data(data_path)
        extreme_events = identify_extreme_events(data)
        extreme_summary = summarize_extreme_events(extreme_events, variable_name)
        combined_extreme_events = merge_extreme_event_data(combined_extreme_events, extreme_summary)
        correlation_results = correlate_with_population(extreme_summary, population)
        print(f"Correlation between extreme events for {file_name} and bird population:")
        print(correlation_results)
    return combined_extreme_events

# Single Responsibility: Merge extreme events summaries
def merge_extreme_event_data(combined_extreme_events, extreme_summary):
    if combined_extreme_events.empty:
        combined_extreme_events = extreme_summary
    else:
        combined_extreme_events = pd.merge(combined_extreme_events, extreme_summary, on='Year', how='outer')
    return combined_extreme_events

# Single Responsibility: Save results to the results folder
def save_results(combined_extreme_events, correlation_matrix, results_folder):
    # Save combined extreme event summary and correlation matrix to the results folder
    combined_extreme_events.to_csv(os.path.join(results_folder, 'combined_extreme_event_summary.csv'), index=False)
    correlation_matrix.to_csv(os.path.join(results_folder, 'climate_variable_population_correlation_matrix.csv'))

# Main function that orchestrates the whole process
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
    
    population = pd.read_csv(population_path).rename(columns={'individualCount': 'Bird Population'})
    
    results_folder = '/home/michael/Education/UoG/Earth Science Master/Thesis/results/'
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    combined_extreme_events = process_climate_data(files, base_data_path, population, results_folder)
    
    population_summary = population.groupby('Year')['Bird Population'].sum().reset_index()
    combined_extreme_events = pd.merge(combined_extreme_events, population_summary, on='Year')
    
    correlation_matrix = combined_extreme_events.drop(columns=['Year']).corr()
    print("\nCorrelation matrix between climate variables and bird population:")
    print(correlation_matrix)
    
    save_results(combined_extreme_events, correlation_matrix, results_folder)
    plot_correlation_matrix(correlation_matrix, results_folder)
    plot_time_series(combined_extreme_events, results_folder)

    print(f"All results and visualizations saved to '{results_folder}'")

if __name__ == "__main__":
    main()
