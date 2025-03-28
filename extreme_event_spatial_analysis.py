import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns
import contextily as ctx
import logging
from typing import Optional, Tuple

# Additional imports for scale bar and font properties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

class ExtremeEventAnalyzer:
    def __init__(
        self,
        data_path: str,
        output_path: str,
        percentile: float = 0.95,
        crs: str = "EPSG:4326",
        logger: Optional[logging.Logger] = None
    ):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.percentile = percentile
        self.crs = crs
        
        self.climate_vars = [
            "air_pressure_values", "air_temperature_values", "wind_values",
            "sea_temp_values", "seawater_level_values", "wave_height_values"
        ]
        self.required_columns = self.climate_vars + ['Year', 'lat', 'lon']
        
        self.df: Optional[pd.DataFrame] = None
        self.extreme_thresholds: Optional[pd.Series] = None
        self.logger = logger or logging.getLogger(__name__)

    def _validate_dataframe(self) -> None:
        """Check if loaded DataFrame contains required columns."""
        missing = set(self.required_columns) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in data: {missing}")

    def load_and_preprocess_data(self) -> None:
        """Loads and preprocesses data with validation and error handling."""
        try:
            self.df = pd.read_csv(self.data_path)
            self._validate_dataframe()
            
            # Data cleaning
            self.df[self.climate_vars] = self.df[self.climate_vars].replace(-1.0, np.nan)
            self.df[self.climate_vars] = self.df[self.climate_vars].apply(pd.to_numeric, errors='coerce')
            
            # Vectorized geometry creation
            self.df['geometry'] = gpd.points_from_xy(self.df.lon, self.df.lat)
            
            # Log basic info
            self.logger.info(f"Data loaded successfully with {len(self.df)} records")
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

    def compute_extreme_event_thresholds(self) -> None:
        """Computes percentile thresholds with missing data handling."""
        try:
            self.extreme_thresholds = self.df[self.climate_vars].quantile(self.percentile)
            self.logger.info("Extreme thresholds computed:\n" + 
                             "\n".join([f"{var}: {val:.2f}" for var, val in self.extreme_thresholds.items()]))
        except KeyError as e:
            self.logger.error(f"Missing climate variables: {str(e)}")
            raise
        except ValueError as e:
            self.logger.error(f"Threshold computation error: {str(e)}")
            raise

    def identify_extreme_events(self) -> None:
        """Identifies extreme events with memory-efficient boolean types."""
        if self.extreme_thresholds is None:
            raise ValueError("Thresholds not computed - run compute_extreme_event_thresholds first")
            
        for var in self.climate_vars:
            extreme_col = f"{var}_extreme"
            self.df[extreme_col] = self.df[var] >= self.extreme_thresholds[var]
            self.df[extreme_col] = self.df[extreme_col].astype('bool')
            
        self.logger.info("Extreme event flags added")

    def aggregate_extreme_events(self) -> pd.DataFrame:
        """Aggregates extreme events with spatial-temporal grouping."""
        extreme_cols = [f"{var}_extreme" for var in self.climate_vars]
        self.df['extreme_event_count'] = self.df[extreme_cols].sum(axis=1).astype('int16')
        
        return self.df.groupby(['Year', 'lat', 'lon'])['extreme_event_count'].sum().reset_index()

    def plot_extreme_event_heatmap(
        self,
        alpha: float = 0.7,
        cmap: str = 'cividis',  # color-blind-friendly colormap
        bw_adjust: Optional[float] = None
    ) -> Path:
        """
        Generates an enhanced heatmap visualization for extreme events.
        
        Args:
            alpha (float): Transparency for the KDE fill.
            cmap (str): Colormap for the KDE. Defaults to a color-blind-friendly palette.
            bw_adjust (Optional[float]): Optional bandwidth adjustment for the KDE plot.
        
        Returns:
            Path: File path to the saved heatmap image.
        """
        # Transform to Web Mercator for basemap compatibility
        gdf = gpd.GeoDataFrame(self.df, geometry='geometry', crs=self.crs).to_crs(epsg=3857)
        
        # Calculate bounds and aspect ratio
        minx, miny, maxx, maxy = gdf.total_bounds
        dx = maxx - minx
        dy = maxy - miny
        aspect_ratio = dy / dx
        
        # Set figure size based on data extent
        fig_width = 12
        fig_height = fig_width * aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Prepare KDE parameters, including optional bandwidth adjustment
        kde_params = {
            'x': gdf.geometry.x,
            'y': gdf.geometry.y,
            'weights': gdf['extreme_event_count'],
            'cmap': cmap,
            'fill': True,
            'alpha': alpha,
            'levels': 20,
            'thresh': 0.05,
            'ax': ax
        }
        if bw_adjust is not None:
            kde_params['bw_adjust'] = bw_adjust
        
        # Create the KDE plot
        kde = sns.kdeplot(**kde_params)
        
        # Set axes limits and ensure equal aspect
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal', adjustable='datalim')
        
        # Add basemap with improved attribution size
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager, attribution_size=8)
        
        # Add a well-styled colorbar
        cb = fig.colorbar(kde.collections[0], ax=ax, orientation='vertical', pad=0.02)
        cb.set_label('Extreme Event Density', fontsize=10)
        cb.ax.tick_params(labelsize=8)
        
        # Enhanced title and an annotation for data source
        ax.set_title(f"Extreme Events Heatmap ({self.percentile*100:.0f}th Percentile)", fontsize=16, pad=15)
        ax.annotate('Data: SMHI Meteorological Institute', xy=(0.01, 0.01),
                    xycoords='axes fraction', fontsize=8, color='gray')
        
        # Optionally add a scale bar (20 km example)
        fontprops = fm.FontProperties(size=8)
        scalebar = AnchoredSizeBar(
            ax.transData,
            80000,  # 80 km in Web Mercator units
            '20 km',
            'lower left',
            pad=0.3,
            color='black',
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops
        )
        ax.add_artist(scalebar)
        
        # Optionally add a simple north arrow
        ax.annotate(
            'N',
            xy=(0.95, 0.05),
            xytext=(0.95, 0.18),
            arrowprops=dict(facecolor='black', width=2, headwidth=8),
            ha='center', va='center',
            fontsize=20,
            xycoords=ax.transAxes
        )
        
        # Remove axes for a cleaner map appearance
        ax.set_axis_off()
        
        # Save the figure
        heatmap_path = self.output_path / 'extreme_events_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Enhanced heatmap saved to {heatmap_path}")
        return heatmap_path

    def run_analysis(self) -> Tuple[pd.DataFrame, Path]:
        """Executes full analysis pipeline with error handling."""
        try:
            self.load_and_preprocess_data()
            self.compute_extreme_event_thresholds()
            self.identify_extreme_events()
            yearly_summary = self.aggregate_extreme_events()
            heatmap_path = self.plot_extreme_event_heatmap()
            
            # Save results
            output_path = self.output_path / 'extreme_events_summary.csv'
            yearly_summary.to_csv(output_path, index=False)
            self.logger.info(f"Analysis complete. Results saved to {output_path}")
            
            return yearly_summary, heatmap_path
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run analyzer
    analyzer = ExtremeEventAnalyzer(
        data_path="/home/michael/Education/UoG/Earth Science Master/Thesis/data/all_bird_data/data_for_correlation.csv",
        output_path="/home/michael/Education/UoG/Earth Science Master/Thesis/results/new_results/extreme_events_analysis_and_population_correlation/",
        percentile=0.95
    )
    
    summary_df, heatmap_file = analyzer.run_analysis()
