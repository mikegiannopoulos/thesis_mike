# Effects of Climate Variability and Extreme Events on Bird Populations along Sweden’s Western Coast

## Overview
This project investigates the relationship between bird populations in western Sweden and various climate variables, including local weather and large-scale teleconnection patterns (e.g., NAO, AAO, Scandinavian Pattern, and East Atlantic Pattern). The project utilizes machine learning, statistical correlation analysis, and factor analysis to identify key drivers of bird population fluctuations.

## Dataset
The project utilizes:
- **Bird Population Data:** From 2015–2022, filtered and processed to remove inconsistencies. https://www.gbif.org/dataset/093d659d-99e1-4bd0-9de7-6330b361ea54
- **Climate Variables:** Data from SMHI (Swedish Meteorological and Hydrological Institute), including air temperature, sea temperature, air pressure, wind speed, wave height, and seawater level. https://www.smhi.se/data/meteorologi/ladda-ner-meteorologiska-observationer/airtemperatureInstant/71420
- **Teleconnection Indices:** North Atlantic Oscillation (NAO), Antarctic Oscillation (AAO), Scandinavian Pattern, and East Atlantic Pattern. https://www.cpc.ncep.noaa.gov/data/teledoc/telecontents.shtml


## Scripts Description

### 1. **`prepare_original_bird_dataset.py`**
Cleans the raw bird population dataset by removing irrelevant columns, applying spatial and temporal filters, and aggregating yearly species counts.

### 2. **`climate_configs.py`**
Defines configurations for loading and processing climate data, including file paths, column names, and metrics for different variables.

### 3. **`bird_population_factor_analysis_regression.py`**
Performs **Exploratory Factor Analysis (EFA)** on climate variables and uses the resulting factors to predict bird populations via linear regression.

### 4. **`bird_population_pca_regression_analysis.py`**
Uses **Principal Component Analysis (PCA)** to reduce dimensionality of climate data and applies regression modeling to predict bird populations for different species.

### 5. **`bird_population_random_forest_and_gradient_boosting.py`**
Implements **Random Forest and Gradient Boosting regression models** to analyze species-specific relationships between bird populations and climate variables.

### 6. **`bird_population_vs_nao_correlation_monthly.py`**
Analyzes the correlation between **monthly** bird population data and teleconnection indices (NAO, AAO, SCAND, EA) using **Pearson and Spearman correlation coefficients**.

### 7. **`bird_population_vs_nao_correlation_daily.py`**
Performs a **daily-scale** correlation analysis of bird population with the NAO and AAO indices, including normality tests and visualization of relationships.

### 8. **`extreme_events_analysis_and_population_correlation.py`**
Identifies extreme climate events (e.g., high temperatures, strong winds) and analyzes their impact on total and species-specific bird populations.

### 9. **`climate_variables_teleconnections_correlation_analysis.py`**
Examines **correlations between local climate variables and teleconnection indices**, including lag correlations to assess delayed effects.

### 10. **`correlation_heatmaps.py`**
Generates heatmaps of correlation matrices for **species-specific climate interactions**, visualizing relationships across multiple variables.

## Usage
To run the scripts, ensure you have the required dependencies installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn factor-analyzer
```

Each script can be executed as:
```bash
python scripts/bird_population_factor_analysis_regression.py
```
Ensure that the data files are located in their respective directories as expected by the scripts.


## Author
**Michael Giannopoulos** - Climate Scientist & Data Analyst

## License
This project is licensed under the MIT License.

