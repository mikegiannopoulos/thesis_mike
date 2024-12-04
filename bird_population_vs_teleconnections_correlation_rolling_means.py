"""
This script performs a correlation analysis between bird population data and multiple climate teleconnection indices
(NAO, AAO, SCAND, EA). The script follows the principles of Single Responsibility (SRP) and Open/Closed Principle (OCP) 
by organizing the code into classes, each handling a specific task, such as data loading, preprocessing, correlation analysis,
and visualization.

The workflow includes the following steps:
1. **Data Loading**: Different loaders are defined to load bird population data and climate index data (NAO, AAO, SCAND, EA) from
CSV files. Each teleconnection index is processed with rolling means and lagging.
2. **Data Preprocessing**: The data is merged to combine bird population data with the climate indices, 
ensuring proper alignment and handling of missing values.
3. **Correlation Analysis**: Pearson and Spearman correlations are calculated between the bird population and 
lagged teleconnection indices. Both methods are implemented, and p-values are calculated to assess statistical significance.
4. **Normality Testing**: The script checks the normality of the bird population data and climate index data using 
the Shapiro-Wilk test.
5. **Visualization**: Relationships between bird population and each climate index are visualized using scatter plots 
with linear regression lines.

### Key Components:
- **DataLoader (abstract class)**: Base class for all data loaders, enforcing a `load_data` method for extensions.
  - `BirdDataLoader`: Loads bird population data.
  - `NAODataLoader`, `AAODataLoader`, `SCANDDataLoader`, `EADataLoader`: Load respective teleconnection index data,
     calculate rolling means and lagged values.
- **DataPreprocessor**: Merges bird population data with teleconnection index data, handling missing values.
- **CorrelationAnalyzer**: Computes Pearson or Spearman correlations between bird population and climate indices.
- **NormalityTester**: Performs Shapiro-Wilk normality test on data columns.
- **Visualizer**: Creates scatter plots of the relationship between bird population and climate indices.
- **BirdPopulationAnalysis**: Orchestrator class that coordinates the entire analysis workflow by loading, merging, analyzing, 
    and visualizing data.

### Key Functions:
- `main()`: The entry point for the script, which initializes data loaders, preprocessor, analyzer, normality tester, 
   and visualizer, then orchestrates the analysis.

### Usage:
To run the analysis, update the file paths in the `main()` function to point to your local data files, and execute the script. 
The results will include:
- Pearson and Spearman correlation coefficients between bird population and lagged climate indices.
- Normality test results for bird population and each teleconnection index.
- Scatter plots showing the relationship between bird population and lagged climate indices.

This script is designed for climate scientists and researchers interested in exploring the effects of large-scale 
teleconnection patterns on bird population trends.

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

# SCAND data loader (SRP)
class SCANDDataLoader(DataLoader):
    def __init__(self, filepath, window_size, lag_period):
        self.filepath = filepath
        self.window_size = window_size
        self.lag_period = lag_period
    
    def load_data(self):
        scand_data = pd.read_csv(self.filepath, parse_dates=['date'], index_col='date')
        scand_data.rename(columns={'scand_index_cdas': 'SCAND_index'}, inplace=True)
        scand_data.sort_index(inplace=True)
        scand_data['SCAND_Rolling_Mean'] = scand_data['SCAND_index'].rolling(window=self.window_size, min_periods=1).mean()
        scand_data['SCAND_Rolling_Mean_Lagged'] = scand_data['SCAND_Rolling_Mean'].shift(self.lag_period)
        return scand_data

# EA data loader (SRP)
class EADataLoader(DataLoader):
    def __init__(self, filepath, window_size, lag_period):
        self.filepath = filepath
        self.window_size = window_size
        self.lag_period = lag_period
    
    def load_data(self):
        ea_data = pd.read_csv(self.filepath, parse_dates=['date'], index_col='date')
        ea_data.rename(columns={'ea_index_cdas': 'EA_Index'}, inplace=True)
        ea_data.sort_index(inplace=True)
        ea_data['EA_Rolling_Mean'] = ea_data['EA_Index'].rolling(window=self.window_size, min_periods=1).mean()
        ea_data['EA_Rolling_Mean_Lagged'] = ea_data['EA_Rolling_Mean'].shift(self.lag_period)
        return ea_data

# Data preprocessor (handles merging, SRP)
class DataPreprocessor:
    @staticmethod
    def merge_data(bird_data, nao_data, aao_data, scand_data, ea_data):
        merged_data = pd.merge_asof(bird_data, nao_data[['NAO_Rolling_Mean', 'NAO_Rolling_Mean_Lagged']], left_index=True, right_index=True)
        merged_data = pd.merge_asof(merged_data, aao_data[['AAO_Rolling_Mean', 'AAO_Rolling_Mean_Lagged']], left_index=True, right_index=True)
        merged_data = pd.merge_asof(merged_data, scand_data[['SCAND_Rolling_Mean', 'SCAND_Rolling_Mean_Lagged']], left_index=True, right_index=True)
        merged_data = pd.merge_asof(merged_data, ea_data[['EA_Rolling_Mean', 'EA_Rolling_Mean_Lagged']], left_index=True, right_index=True)

        # Check for NaN values after merging
        nan_counts = merged_data.isna().sum()
        print("NaN counts after merging:\n", nan_counts)

        # Drop rows with NaNs
        merged_data.dropna(inplace=True)
        
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
        bird_data = self.bird_loader.load_data()
        nao_data = self.nao_loader.load_data()
        aao_data = self.aao_loader.load_data()
        scand_data = self.scand_loader.load_data()
        ea_data = self.ea_loader.load_data()

        # Merge data
        merged_data = self.preprocessor.merge_data(bird_data, nao_data, aao_data, scand_data, ea_data)

        # Perform correlation analysis (Pearson with all indices)
        indices = ['NAO_Rolling_Mean_Lagged', 'AAO_Rolling_Mean_Lagged', 'SCAND_Rolling_Mean_Lagged', 'EA_Rolling_Mean_Lagged']
        for index in indices:
            pearson_corr, pearson_p_value = self.analyzer.perform_correlation(merged_data, 'total_population', index)
            print(f"Pearson correlation with {index}: Coefficient={pearson_corr}, P-value={pearson_p_value}")
        
        # Perform correlation analysis (Spearman with all indices)
        spearman_analyzer = CorrelationAnalyzer(method='spearman')
        for index in indices:
            spearman_corr, spearman_p_value = spearman_analyzer.perform_correlation(merged_data, 'total_population', index)
            print(f"Spearman correlation with {index}: Coefficient={spearman_corr}, P-value={spearman_p_value}")

        # Check normality for total_population and all indices
        stat_pop, p_pop = self.normality_tester.check_normality(merged_data['total_population'])
        print(f"Shapiro-Wilk test for total_population: Statistics={stat_pop}, P-value={p_pop}")

        for index in indices:
            stat, p_value = self.normality_tester.check_normality(merged_data[index])
            print(f"Shapiro-Wilk test for {index}: Statistics={stat}, P-value={p_value}")

        # Visualize relationships with all indices
        for index in indices:
            self.visualizer.visualize_relationship(merged_data, index, 'total_population', f'Bird Population vs. Lagged {index}')

# Main function to orchestrate
def main():
    bird_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/aggregated_bird_data_sorted_for_NAO.csv'
    nao_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/NAO/norm.daily.nao.cdas.z500.20150101_20221231.csv'
    aao_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/AAO/norm.daily.aao.cdas.z700.20150101_20223112.csv'
    scand_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/scand/scand_filtered_2015_2022.csv'
    ea_data_filepath = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/EA/ea_filtered_2015_2022.csv'

    # Initialize the components
    bird_loader = BirdDataLoader(bird_data_filepath)
    nao_loader = NAODataLoader(nao_data_filepath, window_size=30, lag_period=30)
    aao_loader = AAODataLoader(aao_data_filepath, window_size=30, lag_period=30)
    scand_loader = SCANDDataLoader(scand_data_filepath, window_size=30, lag_period=30)
    ea_loader = EADataLoader(ea_data_filepath, window_size=30, lag_period=30)
    
    preprocessor = DataPreprocessor()
    analyzer = CorrelationAnalyzer(method='pearson')  # Start with Pearson
    normality_tester = NormalityTester()
    visualizer = Visualizer()

    # Orchestrate the analysis
    analysis = BirdPopulationAnalysis(bird_loader, nao_loader, aao_loader, scand_loader, ea_loader, preprocessor, analyzer, visualizer, normality_tester)
    analysis.run_analysis()

if __name__ == "__main__":
    main()
