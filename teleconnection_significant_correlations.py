"""
A complete analysis pipeline for processing and analyzing bird species teleconnection relationships.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ------------------
# Configuration Class (SRP)
# ------------------
class AnalysisConfig:
    """
    Configuration for the analysis pipeline.
    """
    def __init__(self, output_dir: str, input_path: str):
        self.output_dir = output_dir
        self.input_path = input_path
        self.season_map = {
            'Winter': [1, 2, 12],
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Autumn': [9, 10, 11]
        }


# ------------------
# Data Processor (SRP)
# ------------------
class DataProcessor:
    """
    Loads and transforms the input data.
    """
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Ensure the input file exists and create output directory if needed."""
        if not os.path.exists(self.config.input_path):
            raise FileNotFoundError(f"Input file {self.config.input_path} not found")
        os.makedirs(self.config.output_dir, exist_ok=True)
        logging.info("Input path validated and output directory ensured.")

    def load_and_transform(self) -> pd.DataFrame:
        """Load CSV and perform melting and pivot transformation."""
        df = pd.read_csv(self.config.input_path)
        logging.info("CSV file loaded.")
        return self._melt_and_split(df)

    def _melt_and_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Melts the data frame and extracts metric components using regex.
        Rows that do not match the expected pattern are dropped with a warning.
        """
        id_vars = ['species']
        value_vars = [col for col in df.columns if col != 'species']
        melted = pd.melt(df, id_vars=id_vars, value_vars=value_vars, 
                         var_name='metric', value_name='value')
        logging.info("DataFrame melted.")

        # Use regex to extract Index, CorrelationType, optional _P marker, and Lag.
        extracted = melted['metric'].str.extract(
            r'^([A-Z]+)_(Pearson|Spearman)(?:_P)?_Lag(\d+)M$'
        )
        # Check if any rows failed to match
        if extracted.isnull().any(axis=1).sum() > 0:
            num_failed = extracted.isnull().any(axis=1).sum()
            logging.warning(f"{num_failed} rows did not match the expected metric pattern and will be dropped.")
            valid = extracted.notnull().all(axis=1)
            melted = melted.loc[valid].reset_index(drop=True)
            extracted = extracted.loc[valid].reset_index(drop=True)

        # Rename extracted columns appropriately
        extracted.columns = ['Index', 'CorrelationType', 'Lag']
        melted = pd.concat([melted, extracted], axis=1)
        melted['Lag'] = melted['Lag'].astype(int)
        melted['Metric'] = melted['metric'].str.contains('_P_').map({True: 'p_value', False: 'coefficient'})
        
        # Pivot table to get p_value and coefficient in separate columns
        pivoted = melted.pivot_table(
            index=['species', 'Index', 'CorrelationType', 'Lag'],
            columns='Metric',
            values='value',
            aggfunc='first'
        ).reset_index()
        logging.info("Data transformation complete.")
        return pivoted


# ------------------
# Statistical Processor (SRP/OCP)
# ------------------
class StatisticalCorrector:
    """
    Applies statistical corrections to the DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_tests = df.shape[0]

    def apply_corrections(self) -> pd.DataFrame:
        """Apply Bonferroni and FDR corrections."""
        self.df = self._add_bonferroni()
        self.df = self._add_fdr()
        return self.df

    def _add_bonferroni(self) -> pd.DataFrame:
        threshold = 0.05 / self.n_tests
        self.df['significant_bonferroni'] = self.df['p_value'] < threshold
        logging.info("Bonferroni correction applied.")
        return self.df

    def _add_fdr(self) -> pd.DataFrame:
        p_values = self.df['p_value'].values
        _, fdr_adjusted_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        self.df['p_value_fdr'] = fdr_adjusted_p
        self.df['significant_fdr'] = self.df['p_value_fdr'] < 0.05
        logging.info("FDR correction applied.")
        return self.df


# ------------------
# Season Mapper (SRP)
# ------------------
class SeasonMapper:
    """
    Adds month and season features based on the 'Lag' field.
    """
    def __init__(self, config: AnalysisConfig):
        self.config = config
        # Build mapping dictionaries once
        self.month_mapping = {i: pd.to_datetime(f'2023-{i}-1').month_name() for i in range(1, 13)}
        self.season_mapping = {}
        for season, months in self.config.season_map.items():
            for m in months:
                self.season_mapping[m] = season

    def add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map the 'Lag' field to month names and seasons using dictionaries.
        """
        df['Month'] = df['Lag'].map(self.month_mapping)
        df['Season'] = df['Lag'].map(self.season_mapping)
        logging.info("Seasonal features added.")
        return df


# ------------------
# Insight Generator (SRP/OCP)
# ------------------
class InsightGenerator:
    """
    Generates insights based on significant correlations.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Expecting 'significant_raw' flag to exist.
        self.significant = df[df['significant_raw']].copy()

    def generate_all_insights(self) -> Dict[str, Any]:
        """Generate all insight summaries."""
        return {
            'seasonal_counts': self._seasonal_counts(),
            'monthly_counts': self._monthly_counts(),
            'species_season': self._species_season()
        }

    def _seasonal_counts(self) -> pd.DataFrame:
        return self.significant.groupby(['Index', 'Season']).size().reset_index(name='counts')

    def _monthly_counts(self) -> pd.DataFrame:
        return self.significant.groupby(['Index', 'Month']).size().reset_index(name='counts')

    def _species_season(self) -> pd.DataFrame:
        return self.significant.groupby(['species', 'Season']).size().reset_index(name='counts')


# ------------------
# Visualization Interface (ISP)
# ------------------
class Visualizer(ABC):
    """
    Abstract base class for visualizations.
    """
    @abstractmethod
    def visualize(self, data: pd.DataFrame, output_path: str) -> None:
        pass


class SeasonalCountVisualizer(Visualizer):
    """
    Visualizes seasonal counts using a barplot.
    """
    def visualize(self, data: pd.DataFrame, output_path: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, x='Index', y='counts', hue='Season')
        plt.title('Significant Correlations by Season and Index')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Seasonal count visualization saved to {output_path}")


class SpeciesSeasonHeatmap(Visualizer):
    """
    Visualizes species-specific seasonal sensitivity as a heatmap.
    """
    def visualize(self, data: pd.DataFrame, output_path: str) -> None:
        heatmap_data = data.pivot(index='species', columns='Season', values='counts').fillna(0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
        plt.title('Species-Specific Seasonal Sensitivity')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Species-season heatmap saved to {output_path}")


# ------------------
# Main Orchestrator (DIP)
# ------------------
class AnalysisPipeline:
    """
    Orchestrates the entire analysis pipeline.
    """
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.data = None
        self.insights = None

    def run(self) -> None:
        """
        Execute all steps: data processing, statistical correction,
        seasonal mapping, insight generation, result saving, and visualization.
        """
        # Data processing
        processor = DataProcessor(self.config)
        self.data = processor.load_and_transform()

        # Statistical corrections
        corrector = StatisticalCorrector(self.data)
        self.data = corrector.apply_corrections()
        # Define a raw significance flag before further analysis.
        self.data['significant_raw'] = self.data['p_value'] < 0.05

        # Seasonal mapping
        season_mapper = SeasonMapper(self.config)
        self.data = season_mapper.add_seasonal_features(self.data)

        # Generate insights
        insight_generator = InsightGenerator(self.data)
        self.insights = insight_generator.generate_all_insights()

        # Save results and generate visualizations
        self._save_results()
        self._generate_visualizations()
        logging.info("Analysis pipeline completed.")

    def _save_results(self) -> None:
        output_path = os.path.join(self.config.output_dir, "significant_correlations.csv")
        significant_data = self.data[self.data['significant_raw']]
        significant_data.to_csv(output_path, index=False)
        logging.info(f"Significant correlations saved to {output_path}")

    def _generate_visualizations(self) -> None:
        visualizations = {
            'seasonal_counts': (SeasonalCountVisualizer(), 'seasonal_counts.png'),
            'species_season': (SpeciesSeasonHeatmap(), 'species_season_heatmap.png')
        }

        for insight_name, (visualizer, filename) in visualizations.items():
            output_path = os.path.join(self.config.output_dir, filename)
            visualizer.visualize(self.insights[insight_name], output_path)


# ------------------
# Execution
# ------------------
if __name__ == '__main__':
    # Define paths (adjust these paths as needed)
    config = AnalysisConfig(
        output_dir="/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/bird_population_vs_nao_correlation_monthly/data_handling",
        input_path="/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/bird_population_vs_nao_correlation_monthly/species_correlation_results.csv"
    )

    pipeline = AnalysisPipeline(config)
    pipeline.run()
    logging.info(f"Analysis complete! Results saved to: {os.path.abspath(config.output_dir)}")