"""
This script analyzes the relationship between bird population data and the North Atlantic Oscillation (NAO) index, 
the Arctic/Antarctic Oscillation (AAO) index, utilizing both Pearson and Spearman correlation techniques. Additionally, 
it checks for normality in the bird population and lagged NAO data, and AAO data, then visualizes their relationship.
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

# Bird data loader (SRP)
class BirdDataLoader(DataLoader):
    def __init__(self, filepath):
        self.filepath = filepath
    
    def load_data(self):
        bird_data = pd.read_csv(self.filepath, parse_dates=['eventDate'], index_col='eventDate')
        return bird_data

# NAO data loader (SRP)
class NAODataLoader(DataLoader):
    def __init__(self, filepath, window_size, lag_period):
        self.filepath = filepath
        self.window_size = window_size
        self.lag_period = lag_period
    
    def load_data(self):
        nao_data = pd.read_csv(self.filepath, parse_dates=['date'], index_col='date')
        nao_data.rename(columns={'nao_index_cdas': 'NAO_Index'}, inplace=True)
        nao_data.sort_index(inplace=True)
        nao_data['NAO_Rolling_Mean'] = nao_data['NAO_Index'].rolling(window=self.window_size, min_periods=1).mean()
        nao_data['NAO_Rolling_Mean_Lagged'] = nao_data['NAO_Rolling_Mean'].shift(self.lag_period)
        return nao_data

# AAO data loader (SRP)
class AAODataLoader(DataLoader):
    def __init__(self, filepath, window_size, lag_period):
        self.filepath = filepath
        self.window_size = window_size
        self.lag_period = lag_period
    
    def load_data(self):
        aao_data = pd.read_csv(self.filepath, parse_dates=['date'], index_col='date')
        aao_data.rename(columns={'aao_index_cdas': 'AAO_Index'}, inplace=True)
        aao_data.sort_index(inplace=True)
        aao_data['AAO_Rolling_Mean'] = aao_data['AAO_Index'].rolling(window=self.window_size, min_periods=1).mean()
        aao_data['AAO_Rolling_Mean_Lagged'] = aao_data['AAO_Rolling_Mean'].shift(self.lag_period)
        return aao_data

# Data preprocessor (handles merging, SRP)
class DataPreprocessor:
    @staticmethod
    def merge_data(bird_data, nao_data, aao_data):
        merged_data = pd.merge_asof(bird_data, nao_data[['NAO_Rolling_Mean', 'NAO_Rolling_Mean_Lagged']], left_index=True, right_index=True)
        merged_data = pd.merge_asof(merged_data, aao_data[['AAO_Rolling_Mean', 'AAO_Rolling_Mean_Lagged']], left_index=True, right_index=True)
        return merged_data

# Correlation analyzer (SRP, open for extension)
class CorrelationAnalyzer:
    def __init__(self, method='pearson'):
        self.method = method
    
    def perform_correlation(self, data, col1, col2):
        if self.method == 'pearson':
            corr, p_value = pearsonr(data[col1], data[col2])
        elif self.method == 'spearman':
            corr, p_value = spearmanr(data[col1], data[col2])
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return corr, p_value

# Normality tester class (SRP)
class NormalityTester:
    @staticmethod
    def check_normality(data_column):
        stat, p_value = shapiro(data_column)
        return stat, p_value

# Visualizer class (SRP)
class Visualizer:
    @staticmethod
    def visualize_relationship(merged_data, x_col, y_col, title):
        sns.lmplot(x=x_col, y=y_col, data=merged_data)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

# Main orchestrator class (SRP)
class BirdPopulationAnalysis:
    def __init__(self, bird_loader, nao_loader, aao_loader, preprocessor, analyzer, visualizer, normality_tester):
        self.bird_loader = bird_loader
        self.nao_loader = nao_loader
        self.aao_loader = aao_loader
        self.preprocessor = preprocessor
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.normality_tester = normality_tester

    def run_analysis(self):
        # Load data
        bird_data = self.bird_loader.load_data()
        nao_data = self.nao_loader.load_data()
        aao_data = self.aao_loader.load_data()

        # Merge data
        merged_data = self.preprocessor.merge_data(bird_data, nao_data, aao_data)

        # Perform correlation analysis (Pearson with NAO)
        pearson_corr_nao, pearson_p_value_nao = self.analyzer.perform_correlation(merged_data, 'total_population', 'NAO_Rolling_Mean_Lagged')
        print(f"Pearson correlation with NAO: Coefficient={pearson_corr_nao}, P-value={pearson_p_value_nao}")

        # Perform correlation analysis (Pearson with AAO)
        pearson_corr_aao, pearson_p_value_aao = self.analyzer.perform_correlation(merged_data, 'total_population', 'AAO_Rolling_Mean_Lagged')
        print(f"Pearson correlation with AAO: Coefficient={pearson_corr_aao}, P-value={pearson_p_value_aao}")

        # Perform correlation analysis (Spearman with NAO)
        spearman_analyzer = CorrelationAnalyzer(method='spearman')
        spearman_corr_nao, spearman_p_value_nao = spearman_analyzer.perform_correlation(merged_data, 'total_population', 'NAO_Rolling_Mean_Lagged')
        print(f"Spearman correlation with NAO: Coefficient={spearman_corr_nao}, P-value={spearman_p_value_nao}")

        # Perform correlation analysis (Spearman with AAO)
        spearman_corr_aao, spearman_p_value_aao = spearman_analyzer.perform_correlation(merged_data, 'total_population', 'AAO_Rolling_Mean_Lagged')
        print(f"Spearman correlation with AAO: Coefficient={spearman_corr_aao}, P-value={spearman_p_value_aao}")

        # Check normality for total_population
        stat_pop, p_pop = self.normality_tester.check_normality(merged_data['total_population'])
        print(f"Shapiro-Wilk test for total_population: Statistics={stat_pop}, P-value={p_pop}")

        # Check normality for NAO_Rolling_Mean_Lagged
        stat_nao, p_nao = self.normality_tester.check_normality(merged_data['NAO_Rolling_Mean_Lagged'])
        print(f"Shapiro-Wilk test for NAO_Rolling_Mean_Lagged: Statistics={stat_nao}, P-value={p_nao}")

        # Check normality for AO_Rolling_Mean_Lagged
        stat_aao, p_aao = self.normality_tester.check_normality(merged_data['AAO_Rolling_Mean_Lagged'])
        print(f"Shapiro-Wilk test for AAO_Rolling_Mean_Lagged: Statistics={stat_aao}, P-value={p_aao}")

        # Visualize relationship with NAO
        self.visualizer.visualize_relationship(merged_data, 'NAO_Rolling_Mean_Lagged', 'total_population', 'Bird Population vs. Lagged NAO Rolling Mean')

        # Visualize relationship with AAO
        self.visualizer.visualize_relationship(merged_data, 'AAO_Rolling_Mean_Lagged', 'total_population', 'Bird Population vs. Lagged AAO Rolling Mean')

# Main function to orchestrate
def main():
    bird_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/aggregated_bird_data_sorted_for_NAO.csv'
    nao_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/NAO/norm.daily.nao.cdas.z500.20150101_20221231.csv'
    aao_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/AAO/norm.daily.aao.cdas.z700.20150101_20223112.csv'

    # Initialize the components
    bird_loader = BirdDataLoader(bird_data_filepath)
    nao_loader = NAODataLoader(nao_data_filepath, window_size=30, lag_period=30)
    aao_loader = AAODataLoader(aao_data_filepath, window_size=30, lag_period=30)
    preprocessor = DataPreprocessor()
    analyzer = CorrelationAnalyzer(method='pearson')  # Start with Pearson
    normality_tester = NormalityTester()
    visualizer = Visualizer()

    # Orchestrate the analysis
    analysis = BirdPopulationAnalysis(bird_loader, nao_loader, aao_loader, preprocessor, analyzer, visualizer, normality_tester)
    analysis.run_analysis()

if __name__ == "__main__":
    main()
    