import pandas as pd

# File paths
INPUT_FILE = "/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/filtered_bird_data_for_pairing.csv"
OUTPUT_FILE = "/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/filtered_bird_data_for_pairing_NAO_sorted.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE, parse_dates=['eventDate'])

# Sort by date
df.sort_values(by='eventDate', inplace=True)

# Drop rows with missing total population
df.dropna(subset=['total_population'], inplace=True)

# Filter species with at least 50 observations
species_counts = df['species'].value_counts()
valid_species = species_counts[species_counts >= 50].index
df = df[df['species'].isin(valid_species)]

# Save cleaned data
df.to_csv(OUTPUT_FILE, index=False)

print(f"Preprocessing complete. Saved cleaned data to {OUTPUT_FILE}")
