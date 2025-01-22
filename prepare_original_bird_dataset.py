import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename):
    """Loads raw data from the file."""
    return pd.read_csv(filename, sep="\t", on_bad_lines="skip")


def clean_data(ds):
    """Cleans the dataset by dropping unnecessary columns and renaming."""
    ds.dropna(axis=1, how="all", inplace=True)

    # Drop irrelevant columns
    col_to_drop = [
        'gbifID', 'occurrenceID', 'recordedBy', 'organismQuantity', 'taxonomicStatus',
        'eventTime', 'eventID', 'startDayOfYear', 'endDayOfYear', 'sampleSizeValue',
        'locationID', 'county', 'taxonID', 'scientificName', 'order', 'family', 'genus',
        'genericName', 'specificEpithet', 'infraspecificEpithet', 'taxonRank',
        'vernacularName', 'lastInterpreted', 'taxonKey', 'acceptedTaxonKey',
        'orderKey', 'familyKey', 'genusKey', 'speciesKey', 'acceptedScientificName',
        'verbatimScientificName', 'lastParsed', 'level0Gid', 'level0Name',
        'level1Gid', 'level1Name', 'level2Gid', 'level2Name', 'year', 'month', 'day',
        'iucnRedListCategory', 'organismQuantityType', 'occurrenceStatus', 'class', 'classKey', 'year'
    ]
    col_to_drop += [x for x in ds.columns if len(ds[x].unique()) == 1]

    ds.rename(columns={"decimalLatitude": "lat", "decimalLongitude": "lon"}, inplace=True)
    ds = ds.drop(columns=col_to_drop)
    return ds


def apply_spatial_filter(ds, lat_min, lat_max, lon_min, lon_max):
    """Filters the dataset by latitude and longitude."""
    ds = ds[
        (ds["lat"] >= lat_min)
        & (ds["lat"] <= lat_max)
        & (ds["lon"] >= lon_min)
        & (ds["lon"] <= lon_max)
    ]
    return ds


def apply_temporal_filter(ds, start_year, end_year):
    """Filters the dataset by year."""
    ds["year"] = pd.to_datetime(ds["eventDate"]).dt.year
    ds = ds[(ds["year"] >= start_year) & (ds["year"] <= end_year)]
    return ds


def extract_yearly_distribution(ds):
    """Extracts the yearly distribution of species populations."""
    species_yearly_counts = (
        ds.groupby(["species", "year"])["individualCount"]
        .sum()  # Sum the individual counts to get total population
        .unstack(fill_value=0)
    )
    return species_yearly_counts


def apply_yearly_population_threshold(species_yearly_counts, min_years, min_count_per_year):
    """
    Filters species based on a yearly population threshold.
    
    Parameters:
    - min_years: Minimum number of years a species must have data.
    - min_count_per_year: Minimum population required per year.
    """
    valid_species = (species_yearly_counts >= min_count_per_year).sum(axis=1) >= min_years
    return species_yearly_counts[valid_species]


def save_data(ds, species_yearly_counts, raw_output_path, aggregated_output_path):
    """Saves the raw data and the yearly aggregated data to CSV files."""
    # Save raw data
    ds.to_csv(raw_output_path, index=False)
    print(f"Raw data saved to {raw_output_path}")

    # Save yearly distribution
    species_yearly_counts.to_csv(aggregated_output_path, index=True)
    print(f"Yearly distribution saved to {aggregated_output_path}")


def plot_yearly_trends(species_yearly_counts, save_path=None):
    """Plots yearly trends for each species and saves the plot."""
    plt.figure(figsize=(14, 8))
    for species in species_yearly_counts.index:
        plt.plot(
            species_yearly_counts.columns,
            species_yearly_counts.loc[species],
            label=species,
            marker="o",
        )
    plt.title("Yearly Trends of Species")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_boxplot(species_yearly_counts, save_path=None):
    """Plots a boxplot for yearly species distribution and saves the plot."""
    boxplot_data = species_yearly_counts.T
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=boxplot_data, orient="h", palette="Set2")
    plt.title("Yearly Count Distribution Per Species")
    plt.xlabel("Population")
    plt.ylabel("Species")
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    # File paths
    filename = "/home/michael/masters_thesis/bird_data/occurrence.txt"
    raw_output_path = "/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/bird_data_filtered.csv"
    aggregated_output_path = "/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/yearly_species_filtered.csv"
    trend_plot_path = "/home/michael/Education/UoG/Earth Science Master/Thesis/plots/species_trends_filtered.png"
    boxplot_path = "/home/michael/Education/UoG/Earth Science Master/Thesis/plots/species_boxplot_filtered.png"

    # Spatial filter parameters (e.g., western Sweden)
    lat_min = 55
    lat_max = 59.5
    lon_min = 10.8
    lon_max = 13.5

    # Temporal filter parameters
    start_year = 2015
    end_year = 2022

    # Yearly population threshold parameters
    min_years = 6  # Minimum 6 years of data
    min_count_per_year = 200  # Minimum 10 individuals per year

    # Load and clean data
    ds = load_data(filename)
    ds = clean_data(ds)

    # Apply spatial and temporal filters
    ds = apply_spatial_filter(ds, lat_min, lat_max, lon_min, lon_max)
    ds = apply_temporal_filter(ds, start_year, end_year)

    # Extract yearly distribution
    species_yearly_counts = extract_yearly_distribution(ds)

    # Apply yearly population threshold
    filtered_species_yearly_counts = apply_yearly_population_threshold(
        species_yearly_counts, min_years, min_count_per_year
    )

    # Keep only rows for valid species in the raw dataset
    valid_species = filtered_species_yearly_counts.index
    filtered_ds = ds[ds["species"].isin(valid_species)]

    # Save filtered raw and aggregated data
    save_data(filtered_ds, filtered_species_yearly_counts, raw_output_path, aggregated_output_path)

    # Plot and save figures
    plot_yearly_trends(filtered_species_yearly_counts, save_path=trend_plot_path)
    plot_boxplot(filtered_species_yearly_counts, save_path=boxplot_path)
