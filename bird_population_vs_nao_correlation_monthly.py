"""
Orchestrate the analysis of bird population data with NAO, AAO, SCAND, and EA indices.

This script loads the bird population data, NAO, AAO, SCAND, and EA indices, merges the data,
performs Pearson and Spearman correlation analyses along with normality tests, and visualizes the results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, shapiro
from abc import ABC, abstractmethod

# Base class for data loaders (Open for extension)
class DataLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass

# Bird data loader
class BirdDataLoader(DataLoader):
    def __init__(self, filepath):
        self.filepath = filepath
    
    def load_data(self):
        bird_data = pd.read_csv(self.filepath, parse_dates=['eventDate'], index_col='eventDate')
        # Resample the bird data to monthly means
        bird_data = bird_data.resample('M').mean()
        return bird_data

# NAO data loader
class NAODataLoader(DataLoader):
    def __init__(self, filepath, lag_period):
        self.filepath = filepath
        self.lag_period = lag_period
    
    def load_data(self):
        nao_data = pd.read_csv(self.filepath, parse_dates=['date'], index_col='date')
        nao_data.rename(columns={'nao_index_cdas': 'NAO_Index'}, inplace=True)
        nao_data.sort_index(inplace=True)
        # Calculate monthly means
        nao_data = nao_data.resample('M').mean()
        # Apply lag period to monthly data
        nao_data['NAO_Mean_Lagged'] = nao_data['NAO_Index'].shift(self.lag_period)
        return nao_data

# AAO data loader
class AAODataLoader(DataLoader):
    def __init__(self, filepath, lag_period):
        self.filepath = filepath
        self.lag_period = lag_period
    
    def load_data(self):
        aao_data = pd.read_csv(self.filepath, parse_dates=['date'], index_col='date')
        aao_data.rename(columns={'aao_index_cdas': 'AAO_Index'}, inplace=True)
        aao_data.sort_index(inplace=True)
        # Calculate monthly means
        aao_data = aao_data.resample('M').mean()
        # Apply lag period to monthly data
        aao_data['AAO_Mean_Lagged'] = aao_data['AAO_Index'].shift(self.lag_period)
        return aao_data

# SCAND data loader
class SCANDDataLoader(DataLoader):
    def __init__(self, filepath, lag_period):
        self.filepath = filepath
        self.lag_period = lag_period
    
    def load_data(self):
        scand_data = pd.read_csv(self.filepath, parse_dates=['date'], index_col='date')
        scand_data.rename(columns={'scand_index_cdas': 'SCAND_Index'}, inplace=True)
        scand_data.sort_index(inplace=True)
        # Calculate monthly means
        scand_data = scand_data.resample('M').mean()
        # Apply lag period to monthly data
        scand_data['SCAND_Mean_Lagged'] = scand_data['SCAND_Index'].shift(self.lag_period)
        return scand_data

# EA data loader
class EADataLoader(DataLoader):
    def __init__(self, filepath, lag_period):
        self.filepath = filepath
        self.lag_period = lag_period
    
    def load_data(self):
        ea_data = pd.read_csv(self.filepath, parse_dates=['date'], index_col='date')
        ea_data.rename(columns={'ea_index_cdas': 'EA_Index'}, inplace=True)
        ea_data.sort_index(inplace=True)
        # Calculate monthly means
        ea_data = ea_data.resample('M').mean()
        # Apply lag period to monthly data
        ea_data['EA_Mean_Lagged'] = ea_data['EA_Index'].shift(self.lag_period)
        return ea_data

# Data preprocessor (handles merging)
class DataPreprocessor:
    @staticmethod
    def merge_data(bird_data, nao_data, aao_data, scand_data, ea_data):
        # Merging data based on the monthly index
        merged_data = pd.merge_asof(bird_data, nao_data[['NAO_Index', 'NAO_Mean_Lagged']], left_index=True, right_index=True)
        merged_data = pd.merge_asof(merged_data, aao_data[['AAO_Index', 'AAO_Mean_Lagged']], left_index=True, right_index=True)
        merged_data = pd.merge_asof(merged_data, scand_data[['SCAND_Index', 'SCAND_Mean_Lagged']], left_index=True, right_index=True)
        merged_data = pd.merge_asof(merged_data, ea_data[['EA_Index', 'EA_Mean_Lagged']], left_index=True, right_index=True)
        merged_data = merged_data.dropna()  # Drop rows with missing values
        return merged_data

# Correlation analyzer (Open for extension)
class CorrelationAnalyzer:
    def perform_correlation(self, data, col1, col2, method='pearson'):
        if method == 'pearson':
            corr, p_value = pearsonr(data[col1], data[col2])
        elif method == 'spearman':
            corr, p_value = spearmanr(data[col1], data[col2])
        else:
            raise ValueError(f"Unknown method: {method}")
        return corr, p_value

# Normality tester class
class NormalityTester:
    @staticmethod
    def check_normality(data_column):
        stat, p_value = shapiro(data_column)
        return stat, p_value

# Visualizer class
class Visualizer:
    @staticmethod
    def visualize_relationship(merged_data, x_col, y_col, title, save_path=None):
        sns.lmplot(x=x_col, y=y_col, data=merged_data, aspect=1.5, height=6)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.show()

# Main orchestrator class with print statements for correlation results
class BirdPopulationAnalysis:
    def __init__(self, bird_loader, nao_loader, aao_loader, scand_loader, ea_loader, preprocessor, analyzer, visualizer, normality_tester):
        self.bird_loader = bird_loader
        self.nao_loader = nao_loader
        self.aao_loader = aao_loader
        self.scand_loader = scand_loader
        self.ea_loader = ea_loader
        self.preprocessor = preprocessor
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.normality_tester = normality_tester

    def run_analysis(self):
        # Load data
        print("Loading data...")
        bird_data = self.bird_loader.load_data()
        nao_data = self.nao_loader.load_data()
        aao_data = self.aao_loader.load_data()
        scand_data = self.scand_loader.load_data()
        ea_data = self.ea_loader.load_data()
        print("Data loaded successfully.\n")

        # Merge data
        print("Merging data...")
        merged_data = self.preprocessor.merge_data(bird_data, nao_data, aao_data, scand_data, ea_data)
        print("Data merged successfully.\n")

        # Validate merged data
        print("Validating merged data:")
        print(merged_data.head())
        print("\nMissing values per column:")
        print(merged_data.isnull().sum())
        print("\n")

        # Dictionary to store results
        results = {
            'Metric': [],
            'Value': [],
            'P-value': []
        }

        # Define indices and their corresponding lagged columns
        indices = {
            'NAO': 'NAO_Mean_Lagged',
            'AAO': 'AAO_Mean_Lagged',
            'SCAND': 'SCAND_Mean_Lagged',
            'EA': 'EA_Mean_Lagged'
        }

        # Define correlation methods
        methods = ['pearson', 'spearman']

        # Perform correlation analyses
        for method in methods:
            for index, label in indices.items():
                corr, p = self.analyzer.perform_correlation(merged_data, 'total_population', label, method=method)
                print(f"{method.capitalize()} correlation with {index}: {corr:.3f}, p-value: {p:.3f}")
                results['Metric'].append(f"{method.capitalize()} correlation with {index}")
                results['Value'].append(corr)
                results['P-value'].append(p)
            print("\n")  # Add space between methods

        # Perform normality tests for each lagged index
        for index, label in indices.items():
            stat, p = self.normality_tester.check_normality(merged_data[label])
            print(f"Shapiro-Wilk test for {label}: Statistic={stat:.3f}, p-value={p:.3f}")
            results['Metric'].append(f"Shapiro-Wilk test for {label}")
            results['Value'].append(stat)
            results['P-value'].append(p)
        print("\n")

        # Convert results dictionary to a DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        output_csv_path = '/home/michael/Education/UoG/Earth Science Master/Thesis/analysis_results.csv'
        results_df.to_csv(output_csv_path, index=False)
        print(f"Analysis results saved to {output_csv_path}\n")

        # Visualize relationships
        print("Generating visualizations...")
        for index, label in indices.items():
            title = f'Bird Population vs. Lagged {index} Mean'
            # save_path = f'/home/michael/Education/UoG/Earth Science Master/Thesis/plots/{index}_relationship.png'
            self.visualizer.visualize_relationship(
                merged_data, 
                x_col=label, 
                y_col='total_population', 
                title=title,
                # save_path=save_path
            )
        print("Visualizations generated successfully.\n")

def main():
    # File paths
    bird_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/aggregated_bird_data_sorted_for_NAO.csv'
    nao_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/NAO/norm.daily.nao.cdas.z500.20150101_20221231.csv'
    aao_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/AAO/norm.daily.aao.cdas.z700.20150101_20223112.csv'
    scand_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/scand/scand_filtered_2015_2022.csv'
    ea_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/EA/ea_filtered_2015_2022.csv'

    # Initialize the components
    bird_loader = BirdDataLoader(bird_data_filepath)
    nao_loader = NAODataLoader(nao_data_filepath, lag_period=3)
    aao_loader = AAODataLoader(aao_data_filepath, lag_period=3)
    scand_loader = SCANDDataLoader(scand_data_filepath, lag_period=3)
    ea_loader = EADataLoader(ea_data_filepath, lag_period=3)
    preprocessor = DataPreprocessor()
    analyzer = CorrelationAnalyzer()
    normality_tester = NormalityTester()
    visualizer = Visualizer()

    # Orchestrate the analysis
    analysis = BirdPopulationAnalysis(
        bird_loader, 
        nao_loader, 
        aao_loader, 
        scand_loader, 
        ea_loader, 
        preprocessor, 
        analyzer, 
        visualizer, 
        normality_tester
    )
    analysis.run_analysis()

if __name__ == "__main__":
    main()
