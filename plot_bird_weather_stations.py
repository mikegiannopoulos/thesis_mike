import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Load dataset
file_path = "/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/paired_birds_all_climate_data.csv"
df = pd.read_csv(file_path)

# Load only necessary columns to reduce memory usage
cols = [col for col in pd.read_csv(file_path, nrows=0).columns 
        if "_nearest_station" in col] + ["lat", "lon"]
df = pd.read_csv(file_path, usecols=cols)

# Optimized weather station processing using vectorized operations
station_series = pd.concat([df[col] for col in cols if "_nearest_station" in col])
unique_stations = station_series.dropna().unique()

if len(unique_stations) > 0:
    split_stations = pd.Series(unique_stations).str.split('_', expand=True)
    split_stations.columns = ['lat_str', 'lon_str']
    coords = (split_stations.apply(pd.to_numeric, errors='coerce')
              .dropna()
              .rename(columns={'lat_str': 'lat', 'lon_str': 'lon'}))
    weather_stations = list(zip(coords['lon'], coords['lat']))
else:
    weather_stations = []

# Create figure with optimized features
fig, ax = plt.subplots(figsize=(12, 10), 
                      subplot_kw={'projection': ccrs.LambertConformal(central_longitude=12.5)})

# Set map extent first for better performance
ax.set_extent([9, 15.5, 55.5, 59.5], crs=ccrs.PlateCarree())

# Add optimized geographical features
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='azure')
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f0f0e6', edgecolor='black')
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='#b0c4de', edgecolor='black')
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.0, edgecolor='black')
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='-', linewidth=0.7, edgecolor='gray')


# Plot bird locations with optimized parameters
bird_plot = ax.scatter(df["lon"], df["lat"], 
                      c='#1f77b4',  # More professional blue color
                      s=200,          # Smaller point size
                      alpha=0.8, 
                      transform=ccrs.PlateCarree(),
                      edgecolors='none',  # Remove edge for cleaner look
                      label="Bird Survey Locations",
                      )

# Plot weather stations if available
if weather_stations:
    station_lons, station_lats = zip(*weather_stations)
    station_plot = ax.scatter(station_lons, station_lats,
                            color='#d62728',  # Contrasting red color
                            marker='^', 
                            s=220,
                            transform=ccrs.PlateCarree(),
                            label="Weather Stations",
                            edgecolor='black',
                            linewidth=0.6)

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

# Add north arrow 
ax.annotate('', 
            xy=(base_lon, tip_lat + 0.05),  # Slightly higher arrow tip
            xytext=(base_lon, base_lat),  
            arrowprops=dict(arrowstyle='->, head_width=0.5', 
                            linewidth=4,  # Thicker
                            color='black'),
            transform=ccrs.PlateCarree())

ax.text(base_lon, tip_lat + 0.08, 'N',  # Higher text position
        ha='center', va='bottom', 
        transform=ccrs.PlateCarree(),
        fontsize=16, fontweight='bold')


# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

# Legend with improved styling
legend = ax.legend(loc='upper left', frameon=True, 
                   facecolor='white', framealpha=0.9,
                   fontsize=12, borderpad=1, handletextpad=0.7)

# Map title 
ax.set_title("Study Area: Bird Survey Locations & Weather Stations\nSwedish West Coast",
           fontsize=16, pad=20, fontweight='semibold')

plt.tight_layout()
plt.savefig('Study_Area_Map.png', dpi=400, bbox_inches='tight')
plt.show()