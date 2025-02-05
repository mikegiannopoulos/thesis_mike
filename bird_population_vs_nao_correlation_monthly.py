import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, shapiro
from abc import ABC, abstractmethod

# File paths
BIRD_DATA_FILE = "/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/filtered_bird_data_for_pairing_NAO_sorted.csv"
OUTPUT_FILE = "/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/bird_population_vs_nao_correlation_monthly/species_correlation_results.csv"

# Climate data file paths
NAO_FILE = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/NAO/norm.daily.nao.cdas.z500.20150101_20221231.csv'
AAO_FILE = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/AAO/norm.daily.aao.cdas.z700.20150101_20223112.csv'
SCAND_FILE = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/scand/scand_filtered_2015_2022.csv'
EA_FILE = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/EA/ea_filtered_2015_2022.csv'

# Base class for data loaders
class DataLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass

# Bird data loader
class BirdDataLoader(DataLoader):
    def __init__(self, filepath):  # ADD THIS METHOD
        self.filepath = filepath
    
    def load_data(self):
        df = pd.read_csv(self.filepath, parse_dates=['eventDate'])
        # Resample without MultiIndex
        df = df.set_index('eventDate').groupby('species').resample('ME')['total_population'].mean().reset_index()
        df = df.sort_values('eventDate')  # Sort by date
        return df.set_index('eventDate')  # Single date index
    
    
# Climate data loader with multiple lags
class ClimateDataLoader(DataLoader):
    def __init__(self, filepath, csv_column_name, abbreviation, max_lag=6):
        self.filepath = filepath
        self.csv_column_name = csv_column_name  # Original column name in CSV
        self.abbreviation = abbreviation        # Desired abbreviation (NAO/AAO/etc)
        self.max_lag = max_lag  # Maximum number of lag months
    
    def load_data(self):
        # Load CSV with original column name
        df = pd.read_csv(self.filepath, parse_dates=['date'])
        
        # Rename to standardized abbreviation
        df.rename(columns={
            'date': 'Date',
            self.csv_column_name: f'{self.abbreviation}_Index'
        }, inplace=True)
        
        # Process dates and resample
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df.set_index('Date', inplace=True)
        df = df.resample('ME').mean()

        # Create multiple lag columns
        for lag in range(1, self.max_lag + 1):
            df[f'{self.abbreviation}_Lagged_{lag}M'] = df[f'{self.abbreviation}_Index'].shift(lag)

        df = df.sort_index()  # Add this before returning
        return df

    
    
# Data merging class
class DataPreprocessor:
    @staticmethod
    def merge_data(bird_data, climate_data_dict):
        # Bird data now has: index=eventDate, columns=['species', 'total_population']
        merged_data = bird_data.copy().sort_index()
        
        for key, climate_df in climate_data_dict.items():
            climate_df = climate_df.sort_index()
            merged_data = pd.merge_asof(merged_data, climate_df, left_index=True, right_index=True)
        
        return merged_data.dropna()

# Correlation analyzer
class CorrelationAnalyzer:
    def perform_correlation(self, data, col1, col2, method='pearson'):
        if method == 'pearson':
            return pearsonr(data[col1], data[col2])
        elif method == 'spearman':
            return spearmanr(data[col1], data[col2])
        else:
            raise ValueError("Unknown method")

# Main analysis class
class BirdPopulationAnalysis:
    def __init__(self, bird_loader, climate_loaders, preprocessor, analyzer):
        self.bird_loader = bird_loader
        self.climate_loaders = climate_loaders
        self.preprocessor = preprocessor
        self.analyzer = analyzer    

    def run_analysis(self):
        # Load bird data (all species, single date index)
        bird_data = self.bird_loader.load_data()  # Now has columns: ['species', 'total_population']
        climate_data = {key: loader.load_data() for key, loader in self.climate_loaders.items()}

        # Merge ALL DATA at once (no species grouping yet)
        merged_data = self.preprocessor.merge_data(bird_data, climate_data)

        # Get max_lag (assuming all climate loaders use same max_lag)
        max_lag = next(iter(self.climate_loaders.values())).max_lag

        results = []

        # Loop through UNIQUE SPECIES in the merged data
        for species in merged_data['species'].unique():
            # Filter data for this species
            species_df = merged_data[merged_data['species'] == species]

            # Skip if no data
            if species_df.empty:
                continue

            species_result = {"species": species}

            # Calculate correlations for all lags and indices
            for key in self.climate_loaders.keys():
                for lag in range(1, max_lag + 1):
                    lag_column = f"{key}_Lagged_{lag}M"

                    # Pearson
                    pearson_corr, pearson_p = self.analyzer.perform_correlation(
                        species_df, "total_population", lag_column, "pearson"
                    )
                    species_result[f"{key}_Pearson_Lag{lag}M"] = pearson_corr
                    species_result[f"{key}_Pearson_P_Lag{lag}M"] = pearson_p

                    # Spearman
                    spearman_corr, spearman_p = self.analyzer.perform_correlation(
                        species_df, "total_population", lag_column, "spearman"
                    )
                    species_result[f"{key}_Spearman_Lag{lag}M"] = spearman_corr
                    species_result[f"{key}_Spearman_P_Lag{lag}M"] = spearman_p

            results.append(species_result)

        # Save results
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f"Results saved to {OUTPUT_FILE}")


import math

class CorrelationVisualizer:
    def __init__(self, csv_file, output_dir):
        """
        Parameters:
          csv_file (str): Path to the CSV file with correlation results.
          output_dir (str): Directory to save the output plots.
        """
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.df = pd.read_csv(csv_file)
    
    def plot_heatmap(self, climate_var, correlation_type="Pearson"):
        """
        Creates a heatmap of correlations for the given climate variable.
        The heatmap has species on the y-axis and lag (in months) on the x-axis.
        
        Parameters:
          climate_var (str): The climate index abbreviation (e.g., "NAO").
          correlation_type (str): Type of correlation ("Pearson" or "Spearman").
        """
        max_lag = 12  # Updated for 12 months lag
        heatmap_data = pd.DataFrame()
        heatmap_data["species"] = self.df["species"]
        for lag in range(1, max_lag + 1):
            col_name = f"{climate_var}_{correlation_type}_Lag{lag}M"
            if col_name in self.df.columns:
                heatmap_data[f'Lag{lag}'] = self.df[col_name]
        heatmap_data = heatmap_data.set_index("species")
        
        plt.figure(figsize=(10, max(heatmap_data.shape[0] * 0.3, 6)))
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0)
        plt.title(f"{climate_var} {correlation_type} Correlations Heatmap")
        plt.xlabel("Lag (Months)")
        plt.ylabel("Species")
        plt.tight_layout()
        output_file = f"{self.output_dir}/{climate_var}_{correlation_type}_heatmap.png"
        plt.savefig(output_file)
        plt.close()
        print(f"Heatmap saved to {output_file}")

    def plot_lineplot_grid(self, species_list, climate_var, correlation_type="Pearson", max_lag=12):
        """
        Creates a grid of line plots for multiple species.
        
        Parameters:
          species_list (list): List of species names to include in the grid.
          climate_var (str): The climate index abbreviation (e.g., "NAO").
          correlation_type (str): Type of correlation ("Pearson" or "Spearman").
          max_lag (int): Maximum lag in months.
        """
        # Sort species alphabetically
        species_list = sorted(species_list)
        num_species = len(species_list)
        n_cols = 3  # Adjust as desired for layout
        n_rows = math.ceil(num_species / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True)
        
        # Add an overall title to the grid
        fig.suptitle(f"{climate_var} {correlation_type} Correlation vs Lag - Line Plot Grid", fontsize=16)
        plt.subplots_adjust(top=0.88)  # Make room for the supertitle
        
        # Flatten axes for easy iteration
        if num_species == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for ax, species in zip(axes, species_list):
            row = self.df[self.df["species"] == species]
            if row.empty:
                ax.set_visible(False)
                continue
            
            lags = list(range(1, max_lag + 1))
            # Get the correlation values for each lag
            correlations = []
            for lag in lags:
                col_name = f"{climate_var}_{correlation_type}_Lag{lag}M"
                correlations.append(row.iloc[0][col_name])
            
            # Draw a dashed horizontal line at y = 0
            ax.axhline(0, color="black", linestyle="--")
            
            # Plot the line segment by segment, splitting if the segment crosses 0.
            for i in range(len(lags) - 1):
                x0, y0 = lags[i], correlations[i]
                x1, y1 = lags[i+1], correlations[i+1]
                
                # Check if the segment crosses zero.
                if (y0 >= 0 and y1 >= 0) or (y0 < 0 and y1 < 0):
                    # Entire segment is on one side of zero.
                    color = "blue" if y0 >= 0 else "red"
                    ax.plot([x0, x1], [y0, y1], marker="o", color=color)
                else:
                    # The segment crosses zero. Calculate the crossing point.
                    x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
                    # Plot from (x0, y0) to (x_cross, 0) with color based on y0.
                    color0 = "blue" if y0 >= 0 else "red"
                    ax.plot([x0, x_cross], [y0, 0], marker="o", color=color0)
                    # Plot from (x_cross, 0) to (x1, y1) with color based on y1.
                    color1 = "blue" if y1 >= 0 else "red"
                    ax.plot([x_cross, x1], [0, y1], marker="o", color=color1)
            
            ax.set_title(species)
            ax.set_xlabel("Lag (Months)")
            ax.set_ylabel("Correlation")
            ax.grid(True)
        
        # Turn off any unused subplots
        for i in range(len(species_list), len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for the supertitle
        output_file = f"{self.output_dir}/{climate_var}_{correlation_type}_lineplot_grid.png"
        plt.savefig(output_file)
        plt.close()
        print(f"Line plot grid saved to {output_file}")

# Run the analysis
def main():
    bird_loader = BirdDataLoader(BIRD_DATA_FILE)
    
    climate_loaders = {
        "NAO": ClimateDataLoader(NAO_FILE, "nao_index_cdas", "NAO", max_lag=12),
        "AAO": ClimateDataLoader(AAO_FILE, "aao_index_cdas", "AAO", max_lag=12),
        "SCAND": ClimateDataLoader(SCAND_FILE, "scand_index_cdas", "SCAND", max_lag=12),
        "EA": ClimateDataLoader(EA_FILE, "ea_index_cdas", "EA", max_lag=12),
    }

    preprocessor = DataPreprocessor()
    analyzer = CorrelationAnalyzer()

    analysis = BirdPopulationAnalysis(bird_loader, climate_loaders, preprocessor, analyzer)
    analysis.run_analysis()

    # Visualization step:
    output_dir = "/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/bird_population_vs_nao_correlation_monthly"
    visualizer = CorrelationVisualizer(OUTPUT_FILE, output_dir)
    
    teleconnections = ["NAO", "AAO", "SCAND", "EA"]
    # Create heatmaps for all teleconnections (Pearson correlations here)
    for climate_var in teleconnections:
        visualizer.plot_heatmap(climate_var, "Pearson")
    
    # Read species list from the results CSV and sort them alphabetically.
    df_results = pd.read_csv(OUTPUT_FILE)
    species_list = df_results["species"].unique().tolist()
    
    # Create a grid of line plots for all teleconnections
    for climate_var in teleconnections:
        visualizer.plot_lineplot_grid(species_list, climate_var, "Pearson", max_lag=12)

if __name__ == "__main__":
    main()