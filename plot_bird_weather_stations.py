import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import time

# Start timer
start_time = time.time()

# Load dataset
file_path = "/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/paired_birds_all_climate_data.csv"
df = pd.read_csv(file_path)

# Load only necessary columns to reduce memory usage
cols = [col for col in df.columns if "_nearest_station" in col] + ["lat", "lon"]
df = pd.read_csv(file_path, usecols=cols)

# ======================================================================
# New Station Processing Logic
# ======================================================================
def extract_station_coords(variables, df):
    """Extract unique station coordinates for specific variables"""
    station_cols = [f"{var}_nearest_station" for var in variables]
    stations = pd.concat([df[col] for col in station_cols if col in df]).dropna().unique()
    
    if len(stations) == 0:
        return []
    
    # Split coordinates and convert to numeric
    split = pd.Series(stations).str.split('_', expand=True)
    coords = split.apply(pd.to_numeric, errors='coerce').dropna()
    return list(zip(coords[1], coords[0]))  # (lon, lat)

# Define variable categories
atmospheric_vars = ['air_pressure', 'air_temperature', 'wind']
oceanographic_vars = ['seawater_level', 'sea_temp', 'wave_height']

# Get coordinates for each category
atmo_coords = extract_station_coords(atmospheric_vars, df)
ocean_coords = extract_station_coords(oceanographic_vars, df)

# Find overlapping stations
atmo_stations = {f"{lat}_{lon}" for lon, lat in atmo_coords}
ocean_stations = {f"{lat}_{lon}" for lon, lat in ocean_coords}
common_stations = atmo_stations & ocean_stations
common_coords = [tuple(map(float, s.split('_')[::-1])) for s in common_stations]

# Remove common stations from individual categories
atmo_coords = [c for c in atmo_coords if f"{c[1]}_{c[0]}" not in common_stations]
ocean_coords = [c for c in ocean_coords if f"{c[1]}_{c[0]}" not in common_stations]

# ======================================================================
# Mapping Section
# ======================================================================
# Create figure
fig, ax = plt.subplots(figsize=(12, 10), 
                      subplot_kw={'projection': ccrs.LambertConformal(central_longitude=12.5)})

# Set map extent
ax.set_extent([9, 15.5, 55.5, 59.5], crs=ccrs.PlateCarree())

# Add geographical features
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#d4ecff')
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f0f0e6', edgecolor='black')
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='#b2e2ed', edgecolor='black')
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.0, edgecolor='black')
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='-', linewidth=0.7, edgecolor='gray')

# Plot bird locations
bird_plot = ax.scatter(df["lon"], df["lat"], 
                      c='#46a308', s=200, alpha=0.8,
                      transform=ccrs.PlateCarree(),
                      edgecolors='none',
                      label="Bird Survey Locations")

# Plot weather stations
station_plots = []
if atmo_coords:
    lon, lat = zip(*atmo_coords)
    station_plots.append(ax.scatter(lon, lat, color='#d62728', marker='^', s=220,
                                   transform=ccrs.PlateCarree(), edgecolor='black',
                                   linewidth=0.6, label="Atmospheric Stations"))

if ocean_coords:
    lon, lat = zip(*ocean_coords)
    station_plots.append(ax.scatter(lon, lat, color='#1f77b4', marker='^', s=220,
                                   transform=ccrs.PlateCarree(), edgecolor='black',
                                   linewidth=0.6, label="Oceanographic Stations"))

if common_coords:
    lon, lat = zip(*common_coords)
    station_plots.append(ax.scatter(lon, lat, color='#ff7f0e', marker='^', s=240,
                                   transform=ccrs.PlateCarree(), edgecolor='black',
                                   linewidth=0.8, label="Both Station Types"))

# Scale bar using projected coordinates
def add_proper_scale_bar(ax, location=(0.1, 0.1), length_km=100):
    """
    Add a proper scale bar with accurate distance representation
    location: tuple (x, y) in axes coordinates (0-1)
    length_km: desired scale bar length in kilometers
    """
    # Get axes coordinates transformation
    tmc = ax.transAxes - ccrs.PlateCarree()._as_mpl_transform(ax)
    
    # Calculate length in degrees at map center
    center_lat = 57.5  # Approximate map center latitude
    km_per_deg = 111.32  # At equator
    adjusted_km_per_deg = km_per_deg * np.cos(np.deg2rad(center_lat))
    length_deg = length_km / adjusted_km_per_deg
    
    # Convert location to data coordinates
    x0, y0 = tmc.transform((location[0], location[1]))
    
    # Create scale bar elements
    bar_x = [x0, x0 + length_deg]
    bar_y = [y0, y0]
    
    # Black and white alternating pattern
    ax.plot(bar_x, bar_y, color='black', linewidth=4, transform=ccrs.PlateCarree())
    for i in range(0, int(length_km), 25):
        pos = x0 + (i/length_km)*length_deg
        ax.plot([pos, pos], [y0-0.05, y0+0.05], 
                color='black', linewidth=1, transform=ccrs.PlateCarree())
    
    # Add text label
    ax.text(x0 + length_deg/2, y0 - 0.2, f'{length_km} km',
           ha='center', va='top', transform=ccrs.PlateCarree(),
           fontsize=10, backgroundcolor='white')

# Add scale bar to lower left corner
add_proper_scale_bar(ax, location=(0.05, 0.1), length_km=100)

# -- Corrected North Arrow using geographic coordinates --
# Choose coordinates for the arrow (adjust as needed based on your map's layout)
base_lon = 15  # Eastern part of the map
base_lat = 59.0  # Northern position
tip_lat = base_lat + 0.2  # Extend northward

# North Arrow
base_lon, base_lat = 15, 59.0
ax.annotate('', xy=(base_lon, base_lat + 0.25), xytext=(base_lon, base_lat),
           arrowprops=dict(arrowstyle='->, head_width=0.5', linewidth=4, color='black'),
           transform=ccrs.PlateCarree())
ax.text(base_lon, base_lat + 0.3, 'N', ha='center', va='bottom',
       transform=ccrs.PlateCarree(), fontsize=16, fontweight='bold')

# Gridlines and legend
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9,
         fontsize=12, borderpad=1, handletextpad=0.7)

ax.set_title("Study Area: Bird Survey Locations & Weather Stations\nSwedish West Coast",
           fontsize=16, pad=20, fontweight='semibold')

plt.tight_layout()

# Execution time tracking
print(f"\nExecution time: {time.time() - start_time:.2f} seconds")

# Save or show plot
plt.savefig('/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/plot_bird_weather_stations/Study_Area_Map.png', dpi=400, bbox_inches='tight')
plt.show()