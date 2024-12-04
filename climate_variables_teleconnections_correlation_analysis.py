"""
Main script for analyzing correlations between weather variables and climate indices.

This script loads weather data from CSV files, merges it with climate index data,
and computes correlations between the weather variables and the climate indices.
The correlations are then plotted using seaborn.

The script also computes correlations with lagged climate indices and plots
the correlations vs. lag days.

The total time taken to run the script is printed at the end.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
import time  # Import the time module

class DataLoader:
    """Class responsible for loading and preprocessing data."""

    @staticmethod
    def load_weather_data(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.resample('D').mean()
        df['Regional_Average'] = df.mean(axis=1)
        return df

    @staticmethod
    def load_nao_data(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.rename(columns={'nao_index_cdas': 'NAO_Index'}, inplace=True)
        return df

    @staticmethod
    def load_aao_data(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.rename(columns={'aao_index_cdas': 'AAO_Index'}, inplace=True)
        return df

    @staticmethod
    def load_scand_data(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.rename(columns={'scand_index_cdas': 'Scand_Index'}, inplace=True)
        return df

    @staticmethod
    def load_ea_data(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.rename(columns={'ea_index_cdas': 'EA_Index'}, inplace=True)
        return df

class CorrelationCalculator:
    """Class responsible for computing correlations."""
    
    @staticmethod
    def compute_correlation(df, variable_name, index_column):
        df = df.dropna()
        if len(df) < 2:
            print(f"\nVariable: {variable_name} - Not enough data for correlation analysis.")
            return
        pearson_corr, pearson_p = pearsonr(df['Regional_Average'], df[index_column])
        spearman_corr, spearman_p = spearmanr(df['Regional_Average'], df[index_column])
        print(f"\nVariable: {variable_name}")
        print(f"Pearson Correlation: {pearson_corr:.4f}, P-value: {pearson_p:.4e}")
        print(f"Spearman Correlation: {spearman_corr:.4f}, P-value: {spearman_p:.4e}")
        return pearson_corr, spearman_corr

    @staticmethod
    def compute_correlation_with_lag(df, variable_name, index_column, max_lag_days=30):
        print(f"\nVariable: {variable_name}")
        results = []
        for lag in range(0, max_lag_days + 1):
            df[f'{index_column}_Lagged'] = df[index_column].shift(lag)
            df_lagged = df.dropna()
            if len(df_lagged) < 2:
                continue
            pearson_corr, pearson_p = pearsonr(df_lagged['Regional_Average'], df_lagged[f'{index_column}_Lagged'])
            results.append({'Lag_Days': lag, 'Pearson_Correlation': pearson_corr, 'P_Value': pearson_p})
        results_df = pd.DataFrame(results)
        return results_df

class Plotter:
    """Class responsible for plotting data."""

    @staticmethod
    def plot_correlation(df, variable_name, index_column):
        sns.lmplot(x=index_column, y='Regional_Average', data=df, line_kws={'color': 'red'})
        plt.title(f'{variable_name.replace("_", " ").title()} vs. {index_column.replace("_", " ").title()}')
        plt.xlabel(index_column.replace('_', ' ').title())
        plt.ylabel(variable_name.replace('_', ' ').title())
        plt.show()

    @staticmethod
    def plot_lag_correlation(results_df, variable_name):
        if not results_df.empty:
            max_corr_row = results_df.loc[results_df['Pearson_Correlation'].abs().idxmax()]
            print(f"Maximum correlation at lag {int(max_corr_row['Lag_Days'])} days:")
            print(f"Pearson Correlation: {max_corr_row['Pearson_Correlation']:.4f}, P-value: {max_corr_row['P_Value']:.4e}")
            plt.figure(figsize=(10, 5))
            plt.plot(results_df['Lag_Days'], results_df['Pearson_Correlation'], marker='o')
            plt.title(f'Correlation vs. Lag Days for {variable_name.replace("_", " ").title()}')
            plt.xlabel('Lag Days')
            plt.ylabel('Pearson Correlation')
            plt.grid(True)
            plt.show()
        else:
            print("Not enough data for lag analysis.")

# Main function to coordinate the entire workflow
def main():
    start_time = time.time()  # Start the timer

    weather_data_dir = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/SMHI/'
    nao_data_path = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/NAO/norm.daily.nao.cdas.z500.20150101_20221231.csv'
    aao_data_path = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/AAO/norm.daily.aao.cdas.z700.20150101_20223112.csv'
    scand_data_path = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/scand/scand_filtered_2015_2022.csv'
    ea_data_path = '/home/michael/Education/UoG/Earth Science Master/Thesis/data/EA/ea_filtered_2015_2022.csv'

    weather_files = {
        'air_pressure': 'air_pressure.csv',
        'air_temperature': 'air_temperature.csv',
        'sea_temp': 'sea_temp.csv',
        'seawater_level': 'seawater_level.csv',
        'wave_height': 'wave_height.csv',
        'wind': 'wind.csv'
    }

    weather_data = {}
    for var, filename in weather_files.items():
        file_path = os.path.join(weather_data_dir, filename)
        weather_data[var] = DataLoader.load_weather_data(file_path)

    nao_data = DataLoader.load_nao_data(nao_data_path)
    aao_data = DataLoader.load_aao_data(aao_data_path)
    scand_data = DataLoader.load_scand_data(scand_data_path)
    ea_data = DataLoader.load_ea_data(ea_data_path)

    merged_data_nao = {}
    merged_data_aao = {}
    merged_data_scand = {}
    merged_data_ea = {}

    for var in weather_data:
        # Merging with NAO data
        merged_df_nao = pd.merge(
            weather_data[var]['Regional_Average'],
            nao_data[['NAO_Index']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        merged_data_nao[var] = merged_df_nao

        # Merging with AAO data
        merged_df_aao = pd.merge(
            weather_data[var]['Regional_Average'],
            aao_data[['AAO_Index']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        merged_data_aao[var] = merged_df_aao

        # Merging with Scandinavian pattern data
        merged_df_scand = pd.merge(
            weather_data[var]['Regional_Average'],
            scand_data[['Scand_Index']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        merged_data_scand[var] = merged_df_scand

        # Merging with East Atlantic pattern data
        merged_df_ea = pd.merge(
            weather_data[var]['Regional_Average'],
            ea_data[['EA_Index']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        merged_data_ea[var] = merged_df_ea

    # Analyze correlations for NAO
    print("\nNAO Analysis:")
    for var in merged_data_nao:
        CorrelationCalculator.compute_correlation(merged_data_nao[var], var, 'NAO_Index')
        Plotter.plot_correlation(merged_data_nao[var], var, 'NAO_Index')
        lag_results_nao = CorrelationCalculator.compute_correlation_with_lag(merged_data_nao[var], var, 'NAO_Index')
        Plotter.plot_lag_correlation(lag_results_nao, var)

    # Analyze correlations for AAO
    print("\nAAO Analysis:")
    for var in merged_data_aao:
        CorrelationCalculator.compute_correlation(merged_data_aao[var], var, 'AAO_Index')
        Plotter.plot_correlation(merged_data_aao[var], var, 'AAO_Index')
        lag_results_aao = CorrelationCalculator.compute_correlation_with_lag(merged_data_aao[var], var, 'AAO_Index')
        Plotter.plot_lag_correlation(lag_results_aao, var)

    # Analyze correlations for Scandinavian pattern
    print("\nScandinavian Pattern Analysis:")
    for var in merged_data_scand:
        CorrelationCalculator.compute_correlation(merged_data_scand[var], var, 'Scand_Index')
        Plotter.plot_correlation(merged_data_scand[var], var, 'Scand_Index')
        lag_results_scand = CorrelationCalculator.compute_correlation_with_lag(merged_data_scand[var], var, 'Scand_Index')
        Plotter.plot_lag_correlation(lag_results_scand, var)

    # Analyze correlations for East Atlantic pattern
    print("\nEast Atlantic Pattern Analysis:")
    for var in merged_data_ea:
        CorrelationCalculator.compute_correlation(merged_data_ea[var], var, 'EA_Index')
        Plotter.plot_correlation(merged_data_ea[var], var, 'EA_Index')
        lag_results_ea = CorrelationCalculator.compute_correlation_with_lag(merged_data_ea[var], var, 'EA_Index')
        Plotter.plot_lag_correlation(lag_results_ea, var)

    end_time = time.time()  # End the timer
    total_time = end_time - start_time
    print(f"\nTotal time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()

