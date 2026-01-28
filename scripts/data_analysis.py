"""
Exploratory Data Analysis Module for Water Quality Data
Provides comprehensive EDA with statistics, visualizations, and automated reporting.
"""

import importlib
import os
from types import ModuleType

import dask.dataframe as dd
import numpy as np
import pandas as pd
import matplotlib
from seaborn import color_palette

os.environ.setdefault("MPLBACKEND", "Agg")
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import json
import gc  # Garbage collection for memory management

# Import pydantic enhancements from local module
from pydantic_enhancements import (
    ColumnNameValidator,
    NumericStatistics,
    CategoricalStatistics,
    DatasetStatistics,
    DataValidationError,
    ColumnValidationError,
    RangeValidationError,
    DataQualityError,
    ErrorHandler,
    ApplicationSettings,
    load_settings,
    validate_dataset_structure,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Data file path for analysis
DATA_FILE_PATH = Path(__file__).parent.parent / "export" / "cleaned_data" / "cleaned_water_data.csv"


def _import_or_raise(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Required dependency '{module_name}' is not installed. "
            "Please install it in your environment before running the analysis."
        ) from exc


plt = _import_or_raise("matplotlib.pyplot")
sns = _import_or_raise("seaborn")


class EDAConfig(BaseModel):
    """
    Configuration for Exploratory Data Analysis with Pydantic validation.

    Think of this like a **photographer's camera settings** that tells us exactly how to capture the data.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Data loading configuration
    filepath: str = Field(
        default=str(DATA_FILE_PATH),
        description="Path to the cleaned data CSV file"
    )
    chunk_size: Optional[int] = Field(
        default=None,
        description="Chunk size for reading large files (None for auto)"
    )
    max_memory_mb: int = Field(
        default=500,
        description="Maximum memory usage in MB before switching to chunked loading"
    )
    required_columns: List[str] = Field(
        default=['Latitude', 'Longitude', 'Water company', 'River Basin District'],
        description="Columns that must be present in the dataset"
    )
    usecols: Optional[List[str]] = Field(
        default=None,
        description="List of column names to read from the dataset (None for all columns)"
    )

    # Output configuration
    output_dir: str = Field(
        default='reports',
        description="Directory to save the output reports"
    )
    report_format: str = Field(
        default='json',
        description="Format to save the output reports (json, html, markdown)"
    )
    report_name: str = Field(
        default='eda_report',
        description="Name of the output report file"
    )
    save_plots: bool = Field(
        default=True,
        description="Save visualizations as images"
    )
    plot_format: str = Field(
        default='png',
        description="Format to save the plots (png, jpg, pdf, svg)"
    )
    plot_dpi: int = Field(
        default=300,
        description="Resolution for saved plots"
    )

    # Visual Configuration
    figure_size: Tuple[float, float] = Field(
        default=(10, 8),
        description="Size of the figures in inches (width, height)"
    )
    font_size: float = Field(
        default=12,
        description="Base font size for all plots"
    )
    palette: str = Field(
        default='viridis',
        description="Seaborn palette for color schemes"
    )
    style: str = Field(
        default='whitegrid',
        description="Seaborn style for plot appearance"
    )
    grid_style: str = Field(
        default='--',
        description="Grid style for plots (none, --, :, -., etc.)"
    )
    grid_color: str = Field(
        default='lightgray',
        description="Color of the grid lines"
    )

    # Analysis Parameters
    top_n_companies: int = Field(
        default=10,
        description="Number of companies to include in the analysis"
    )
    top_n_rivers: int = Field(
        default=10,
        description="Number of rivers to include in the analysis"
    )
    top_n_spills: int = Field(
        default=10,
        description="Number of spills to include in the analysis"
    )
    top_n_sites: int = Field(
        default=10,
        description="Number of sites to include in the analysis"
    )
    top_n_years: int = Field(
        default=10,
        description="Number of years to include in the analysis"
    )
    top_n_regions: int = Field(
        default=10,
        description="Number of regions to include in the analysis"
    )
    histogram_bins: int = Field(
        default=30,
        description="Number of bins for histograms"
    )
    boxplot_showfliers: bool = Field(
        default=True,
        description="Show outliers in boxplots"
    )
    boxplot_showmeans: bool = Field(
        default=False,
        description="Show means in boxplots"
    )
    outlier_percentile: float = Field(
        default=99.0,
        description="Percentile for outlier detection (0-100, e.g., 99.0 for 99th percentile)"
    )
    outlier_std_threshold: float = Field(
        default=3.0,
        description="Standard deviation threshold for outlier detection"
    )
    missing_value_threshold: float = Field(
        default=0.1,
        description="Threshold for missing values (0-1)"
    )
    duplicate_value_threshold: float = Field(
        default=0.1,
        description="Threshold for duplicate values (0-1)"
    )

    # Data Validation
    optional_columns: List[str] = Field(
        default=['Site Name', 'Receiving Environment', 'Permit Number'],
        description="Optional columns to use if present"
    )
    coordinate_columns: List[str] = Field(
        default=['Latitude', 'Longitude'],
        description="Geographic coordinate columns"
    )
    spill_year_columns: List[str] = Field(
        default=[
            'Spill Events 2020', 'Spill Events 2021', 'Spill Events 2022',
            'Spill Events 2023', 'Spill Events 2024', 'Spill Events 2025'
        ],
        description="Spill event columns by year"
    )
    text_columns: List[str] = Field(
        default=['Water Company', 'River Basin District', 'Site Name', 'Receiving Environment'],
        description="Text columns to validate"
    )
    numeric_columns: List[str] = Field(
        default=['Latitude', 'Longitude'],
        description="Numeric columns for outlier detection"
    )
    lat_min: float = Field(
        default=-90.0,
        description="Minimum valid latitude"
    )
    lat_max: float = Field(
        default=90.0,
        description="Maximum valid latitude"
    )
    lon_min: float = Field(
        default=-180.0,
        description="Minimum valid longitude"
    )
    lon_max: float = Field(
        default=180.0,
        description="Maximum valid longitude"
    )
    spill_year_min: int = Field(
        default=2020,
        description="Minimum valid spill year"
    )
    spill_year_max: int = Field(
        default=2025,
        description="Maximum valid spill year"
    )

    # Report Configuration
    generate_html_report: bool = Field(
        default=True,
        description="Generate HTML report"
    )
    generate_markdown_report: bool = Field(
        default=False,
        description="Generate Markdown report"
    )
    generate_json_report: bool = Field(
        default=True,
        description="Generate JSON report"
    )
    report_title: str = Field(
        default='Water Quality Exploratory Analysis',
        description="Title of the report"
    )
    report_author: str = Field(
        default='Data Analyst',
        description="Author of the report"
    )
    report_date: Optional[datetime] = Field(
        default=None,
        description="Date of the report"
    )

    # Performance Optimization
    max_plot_points: int = Field(
        default=10000,
        description="Maximum number of points to plot"
    )
    max_plot_size: Tuple[int, int] = Field(
        default=(1000, 1000),
        description="Maximum size of the plot in pixels"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of plots"
    )

    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("chunk_size must be greater than 0")
        return v

    @field_validator('max_memory_mb')
    @classmethod
    def validate_max_memory(cls, v):
        if v <= 0 or v > 10000:
            raise ValueError("max_memory_mb must be between 1 and 10000")
        return v

    @field_validator('outlier_percentile')
    @classmethod
    def validate_outlier_percentile(cls, v):
        if v < 0.0 or v > 100.0:
            raise ValueError("outlier_percentile must be between 0 and 100")
        return v

    def model_post_init(self, __context):
        """Set default report date after initialization."""
        if self.report_date is None:
            object.__setattr__(self, 'report_date', datetime.now())


@dataclass
class EDAResults:
    """
    Container for EDA results with structured data.

    Like a **filing cabinet** that organizes all your research findings:
    - Statistics in one folder
    - Charts in another folder
    - Notes and observations in a third folder
    """
    timestamp: str
    company_stats: pd.DataFrame
    regional_stats: pd.DataFrame
    trend_analysis: Dict[str, Any]
    summary_metrics: Dict[str, Any]
    plot_paths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'company_stats': self.company_stats.to_dict('records') if self.company_stats is not None else [],
            'regional_stats': self.regional_stats.to_dict('records') if self.regional_stats is not None else [],
            'trend_analysis': self.trend_analysis,
            'summary_metrics': self.summary_metrics,
            'plot_paths': self.plot_paths,
            'warnings': self.warnings,
            'processing_time_seconds': self.processing_time_seconds
        }

    def save_to_json(self, filepath: str) -> None:
        """Save results to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.info(f"EDA results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save EDA results: {e}")


@contextmanager
def managed_figure(figsize: Tuple[float, float], dpi: int = 100):
    """
    Context manager for matplotlib figures to prevent memory leaks.

    Think of this like a **smart light switch** that:
    - Turns on the light when you enter the room (create figure)
    - Automatically turns off when you leave (closes figure)
    - Even if you trip and fall (error occurs), it still turns off.

    This prevents "forgetting to close figures, which would waste computer memory.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    try:
        yield fig
    finally:
        plt.close(fig)  # Always close, even if error occurs
        gc.collect()  # Clean up memory


def load_data_with_dask(filepath: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load data using Dask for efficient memory management with large files.

    Args:
        filepath: Path to the CSV file
        usecols: Optional list of columns to load

    Returns:
        pandas DataFrame with loaded data
    """
    logger.info(f"Loading data from {filepath} using Dask...")
    try:
        # Use Dask for memory-efficient loading
        ddf = dd.read_csv(filepath, usecols=usecols, assume_missing=True)
        df = ddf.compute()
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data with Dask: {e}")
        raise


class WaterQualityEDA:
    """
    Production-ready exploratory analysis for water quality data.

    Think of this like a **professional research lab** that:
    - Has organized workstations (separate methods for each analysis).
    - Keeps detailed lab notebooks (logging)
    - Cleans up after experiments (memory management)
    - Publishes peer-reviewed papers (automated reports)
    """

    def __init__(self, config: Optional[EDAConfig] = None):
        """Initialize EDA analyzer with configuration."""
        self.config = config or EDAConfig()
        self.error_handler = ErrorHandler()
        self._setup_environment()

    def _setup_environment(self) -> None:
        """
        Set up analysis environment (directories, styles, etc).

        Like preparing a laboratory before experiments - making sure
        all equipment is ready, workspace is clean, and all data is organized.
        """
        # Create output directories
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up styles
        sns.set_style(self.config.style)
        sns.set_palette(self.config.palette)

        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['font.size'] = self.config.font_size
        plt.rcParams['grid.linestyle'] = self.config.grid_style

        # Configure matplotlib for production
        matplotlib.rcParams['figure.max_open_warning'] = 100
        matplotlib.rcParams['agg.path.chunksize'] = 10000  # Optimize for large plots

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file using Dask for efficient memory management.

        Args:
            filepath: Path to the CSV file (defaults to config filepath)

        Returns:
            pandas DataFrame with loaded data
        """
        filepath = filepath or self.config.filepath
        return load_data_with_dask(filepath, self.config.usecols)

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate that DataFrame has required columns and data types.

        Like a teacher checking your homework to make sure you:
        1. Did write something (not empty)
        2. Answered all the required columns (no missing values)
        3. Wrote neatly enough to read (data is valid)
        """
        # Use pydantic enhancements for column validation
        validation_results = validate_dataset_structure(
            df.columns.tolist(),
            required_patterns=[r'^Water Company$', r'^River Basin District$']
        )

        if validation_results.get('warnings'):
            for warning in validation_results['warnings']:
                self.error_handler.add_warning(f"Column warning: {warning}")

        missing_cols = [col for col in self.config.required_columns if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required columns for EDA: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )

        if df.empty:
            raise ValueError("Cannot perform EDA on empty DataFrame")

        logger.info(f"Data validated: {len(df)} rows, {len(df.columns)} columns")

    def calculate_company_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive statistics by water company.

        Like creating a **report card for each company** showing:
        - Average score (mean spills per site)
        - Total homework turned in (total spills)
        - Number of students (site count)
        """
        logger.info("Calculating company statistics...")

        # Check for column name variations
        company_col = 'Water Company' if 'Water Company' in df.columns else 'Water company'
        spill_col = 'Avg_Annual_Spills' if 'Avg_Annual_Spills' in df.columns else None

        if spill_col is None:
            # Try to compute average annual spills from yearly columns
            spill_cols = [col for col in df.columns if 'Spill Events' in col]
            if spill_cols:
                df['Avg_Annual_Spills'] = df[spill_cols].mean(axis=1)
                spill_col = 'Avg_Annual_Spills'
            else:
                logger.warning("No spill data columns found. Using placeholder values.")
                df['Avg_Annual_Spills'] = 0
                spill_col = 'Avg_Annual_Spills'

        stats = df.groupby(company_col)[spill_col].agg([
            ('avg_spills_per_site', 'mean'),
            ('median_spills_per_site', 'median'),
            ('total_spills', 'sum'),
            ('site_count', 'count'),
            ('std_spills', 'std'),
            ('min_spills', 'min'),
            ('max_spills', 'max')
        ]).reset_index()

        # Rename the company column for consistency
        stats = stats.rename(columns={company_col: 'Water company'})

        # Round for readability
        numerical_columns = stats.select_dtypes(include=[np.number]).columns
        stats[numerical_columns] = stats[numerical_columns].round(2)

        # Sort by average spills (descending)
        stats = stats.sort_values(by='avg_spills_per_site', ascending=False)

        logger.info(f"Calculated statistics for {len(stats)} companies")
        return stats

    def calculate_regional_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive statistics by regional district.

        Like comparing different neighborhoods to see which ones
        have more problems (high spills) and which are doing well.
        """
        logger.info("Calculating regional statistics...")

        # Check for column name variations
        region_col = 'River Basin District' if 'River Basin District' in df.columns else None
        spill_col = 'Avg_Annual_Spills' if 'Avg_Annual_Spills' in df.columns else None

        if region_col is None:
            logger.warning("River Basin District column not found")
            return pd.DataFrame()

        if spill_col is None:
            spill_cols = [col for col in df.columns if 'Spill Events' in col]
            if spill_cols:
                df['Avg_Annual_Spills'] = df[spill_cols].mean(axis=1)
                spill_col = 'Avg_Annual_Spills'
            else:
                df['Avg_Annual_Spills'] = 0
                spill_col = 'Avg_Annual_Spills'

        stats = df.groupby(region_col)[spill_col].agg([
            ('avg_spills_per_site', 'mean'),
            ('median_spills_per_site', 'median'),
            ('total_spills', 'sum'),
            ('site_count', 'count'),
            ('std_spills', 'std'),
            ('min_spills', 'min'),
            ('max_spills', 'max')
        ]).reset_index()

        # Rename the region column for consistency
        stats = stats.rename(columns={region_col: 'River Basin District'})

        # Round for readability
        numerical_columns = stats.select_dtypes(include=[np.number]).columns
        stats[numerical_columns] = stats[numerical_columns].round(2)

        # Sort by average spills (descending)
        stats = stats.sort_values('avg_spills_per_site', ascending=False)

        logger.info(f"Calculated statistics for {len(stats)} regions.")
        return stats

    def analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trends in spill data over time.

        Like tracking your grades over the school year to see if
        you're improving (grades going up) or need more study time (grades going down).
        """
        logger.info("Analysing trends...")

        trend_analysis = {
            'overall_trend': 0,
            'improving_sites': 0,
            'worsening_sites': 0,
            'stable_sites': 0,
            'trend_available': False
        }

        # Check if Spill_Trend column exists
        if 'Spill_Trend' in df.columns:
            trend_analysis.update({
                'overall_trend': float(df['Spill_Trend'].mean()),
                'improving_sites': int((df['Spill_Trend'] > 0).sum()),
                'worsening_sites': int((df['Spill_Trend'] < 0).sum()),
                'stable_sites': int((df['Spill_Trend'] == 0).sum()),
                'trend_available': True
            })

            logger.info(f"Trend analysis: {trend_analysis['improving_sites']} improving, "
                        f"{trend_analysis['worsening_sites']} worsening")
        else:
            # Try to calculate trend from yearly spill columns
            spill_year_cols = [col for col in df.columns if 'Spill Events' in col]
            spill_year_cols = sorted(spill_year_cols)  # Sort by year

            if len(spill_year_cols) >= 2:
                logger.info(f"Calculating trends from columns: {spill_year_cols}")
                
                # Calculate trend as difference between last and first year
                first_year_col = spill_year_cols[0]
                last_year_col = spill_year_cols[-1]
                
                # Fill NaN with 0 for calculation
                first_year = df[first_year_col].fillna(0)
                last_year = df[last_year_col].fillna(0)
                
                # Trend: negative means improving (fewer spills), positive means worsening
                df['Spill_Trend'] = last_year - first_year
                
                # Improving = fewer spills (negative trend)
                # Worsening = more spills (positive trend)
                # Stable = no change
                trend_analysis.update({
                    'overall_trend': float(df['Spill_Trend'].mean()),
                    'improving_sites': int((df['Spill_Trend'] < 0).sum()),  # Fewer spills = improving
                    'worsening_sites': int((df['Spill_Trend'] > 0).sum()),  # More spills = worsening
                    'stable_sites': int((df['Spill_Trend'] == 0).sum()),
                    'trend_available': True,
                    'first_year': first_year_col,
                    'last_year': last_year_col
                })

                logger.info(f"Trend analysis ({first_year_col} → {last_year_col}): "
                            f"{trend_analysis['improving_sites']} improving, "
                            f"{trend_analysis['worsening_sites']} worsening, "
                            f"{trend_analysis['stable_sites']} stable")
            else:
                logger.warning("Not enough yearly spill columns to calculate trends.")
        
        return trend_analysis

    def calculate_summary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate overall summary metrics.

        Like writing an executive summary - giving the big picture
        numbers that executives care about (totals, average, maximums).
        """
        logger.info("Calculating summary metrics...")

        # Check for column name variations
        company_col = 'Water Company' if 'Water Company' in df.columns else 'Water company'
        region_col = 'River Basin District' if 'River Basin District' in df.columns else None

        # Ensure Avg_Annual_Spills exists
        if 'Avg_Annual_Spills' not in df.columns:
            spill_cols = [col for col in df.columns if 'Spill Events' in col]
            if spill_cols:
                df['Avg_Annual_Spills'] = df[spill_cols].mean(axis=1)
            else:
                df['Avg_Annual_Spills'] = 0

        metrics = {
            'total_sites': int(len(df)),
            'total_companies': int(df[company_col].nunique()) if company_col in df.columns else 0,
            'total_regions': int(df[region_col].nunique()) if region_col and region_col in df.columns else 0,
            'spills': {
                'mean': float(df['Avg_Annual_Spills'].mean()),
                'median': float(df['Avg_Annual_Spills'].median()),
                'std': float(df['Avg_Annual_Spills'].std()),
                'min': float(df['Avg_Annual_Spills'].min()),
                'max': float(df['Avg_Annual_Spills'].max()),
                'total': int(df['Avg_Annual_Spills'].sum()),
                'percentile_25': float(df['Avg_Annual_Spills'].quantile(0.25)),
                'percentile_75': float(df['Avg_Annual_Spills'].quantile(0.75)),
                'percentile_90': float(df['Avg_Annual_Spills'].quantile(0.90)),
                'percentile_95': float(df['Avg_Annual_Spills'].quantile(0.95)),
                'percentile_99': float(df['Avg_Annual_Spills'].quantile(0.99)),
            },
            'trends': {
                'overall_trend': 0.0,
                'improving_sites': 0,
                'worsening_sites': 0,
                'stable_sites': 0,
                'trend_available': False
            }
        }

        if 'Spill_Trend' in df.columns:
            metrics['trends'].update({
                'overall_trend': float(df['Spill_Trend'].mean()),
                'improving_sites': int((df['Spill_Trend'] < 0).sum()),  # Fewer spills = improving
                'worsening_sites': int((df['Spill_Trend'] > 0).sum()),  # More spills = worsening
                'stable_sites': int((df['Spill_Trend'] == 0).sum()),
                'trend_available': bool(df['Spill_Trend'].notna().any())
            })

        return metrics

    def create_company_visualization(
        self,
        company_stats: pd.DataFrame,
        output_path: str
    ) -> str:
        """
        Create bar chart visualization for companies

        Like making a **racing leaderboard** showing which companies
        have the most spills (top performers) and which are doing worst (bottom performers).
        """
        logger.info(f"Creating company visualization for top {self.config.top_n_companies} companies...")

        # Select top N companies (already sorted by avg_spills_per_site descending)
        top_companies = company_stats.head(self.config.top_n_companies)

        with managed_figure(
            figsize=(self.config.figure_size[0], self.config.figure_size[1]),
            dpi=self.config.plot_dpi
        ) as fig:
            ax = plt.gca()

            # Create horizontal bar chart (easier to read company names)
            sns.barplot(
                data=top_companies,
                y='Water company',
                x='avg_spills_per_site',
                ax=ax,
                palette=self.config.palette
            )

            ax.set_title(
                f'Top {self.config.top_n_companies} Companies by Average Annual Spills',
                fontsize=16,
                fontweight='bold',
                pad=20
            )
            ax.set_xlabel('Average Annual Spills per Site', fontsize=14)
            ax.set_ylabel('Water company', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, linestyle=self.config.grid_style, alpha=0.7)

            # Add data labels (values on top of bars)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', padding=3)

            plt.tight_layout()

            if self.config.save_plots:
                fig.savefig(output_path, bbox_inches='tight', dpi=self.config.plot_dpi)
                logger.info(f"Company visualization saved to {output_path}")

        return output_path

    def create_regional_visualization(
        self,
        regional_stats: pd.DataFrame,
        output_path: str
    ) -> str:
        """
        Create bar chart visualizations for regions.

        Like making a **map** showing which neighborhoods have more
        problems - helps identify where to focus cleanup efforts.
        """
        logger.info("Creating regional visualization...")
        top_regions = regional_stats.head(self.config.top_n_regions)

        with managed_figure(
            figsize=(self.config.figure_size[0], self.config.figure_size[1]),
            dpi=self.config.plot_dpi
        ) as fig:
            ax = plt.gca()

            # Create horizontal bar chart (easier to read region names)
            sns.barplot(
                data=top_regions,
                y='River Basin District',
                x='avg_spills_per_site',
                ax=ax,
                palette=self.config.palette
            )

            ax.set_title(
                f'Top {self.config.top_n_regions} Regions by Average Annual Spills',
                fontsize=16,
                fontweight='bold',
                pad=20
            )
            ax.set_xlabel('Average Annual Spills per Site', fontsize=14)
            ax.set_ylabel('River Basin District', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, linestyle=self.config.grid_style, alpha=0.7)

            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', padding=3)

            plt.tight_layout()

            if self.config.save_plots:
                fig.savefig(output_path, bbox_inches='tight', dpi=self.config.plot_dpi)
                logger.info(f"Regional visualization saved to {output_path}")

        return output_path

    def create_distribution_visualization(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> str:
        """
        Create histogram showing distribution of spills.

        Like making a **height chart for a classroom** - shows how many
        students are short, medium, or tall. Here we show how many sites
        have different spill levels.
        """
        logger.info("Creating distribution visualization...")

        # Ensure Avg_Annual_Spills exists
        if 'Avg_Annual_Spills' not in df.columns:
            spill_cols = [col for col in df.columns if 'Spill Events' in col]
            if spill_cols:
                df['Avg_Annual_Spills'] = df[spill_cols].mean(axis=1)
            else:
                df['Avg_Annual_Spills'] = 0

        # Handle outliers by using percentile cutoff
        cutoff = np.percentile(df['Avg_Annual_Spills'], self.config.outlier_percentile)
        plot_data = df[df['Avg_Annual_Spills'] <= cutoff]['Avg_Annual_Spills']

        outliers_removed = len(df) - len(plot_data)
        if outliers_removed > 0:
            logger.info(
                f"Showing distribution up to {self.config.outlier_percentile}th percentile "
                f"({outliers_removed} extreme outliers excluded from plot)"
            )

        with managed_figure(
            figsize=(self.config.figure_size[0], self.config.figure_size[1]),
            dpi=self.config.plot_dpi
        ) as fig:
            ax = plt.gca()

            # Create histogram with density plot
            ax.hist(
                plot_data,
                bins=self.config.histogram_bins,
                alpha=0.7,
                edgecolor='black',
                color=sns.color_palette(self.config.palette)[0]
            )

            # Add vertical lines for mean and median
            mean_val = df['Avg_Annual_Spills'].mean()
            median_val = df['Avg_Annual_Spills'].median()

            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')

            ax.set_title(
                'Distribution of Annual Spills per Site',
                fontsize=16,
                fontweight='bold',
                pad=20
            )
            ax.set_xlabel('Annual Spills per Site', fontsize=12)
            ax.set_ylabel('Frequency (Number of Sites)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if self.config.save_plots:
                fig.savefig(output_path, bbox_inches='tight', dpi=self.config.plot_dpi)
                logger.info(f"Distribution visualization saved to {output_path}")

        return output_path

    def create_trend_visualization(
        self,
        trend_analysis: Dict[str, Any],
        output_path: str,
    ) -> Optional[str]:
        """
        Create pie chart showing trend distribution.

        Like a **pie chart of favourite ice cream flavors** - shows
        what percentage like chocolate (improving), vanilla (worsening),
        or strawberry (stable).
        """
        if not trend_analysis['trend_available']:
            logger.warning("Trend data not available. Skipping trend visualization.")
            return None

        logger.info("Creating trend visualization...")

        with managed_figure(
            figsize=(self.config.figure_size[0], self.config.figure_size[1]),
            dpi=self.config.plot_dpi
        ) as fig:
            ax = plt.gca()

            labels = ['Improving', 'Worsening', 'Stable']
            values = [
                trend_analysis['improving_sites'],
                trend_analysis['worsening_sites'],
                trend_analysis['stable_sites']
            ]
            colors = ['#2ecc71', '#e74c3c', '#9b59b6']  # Green, Red, Purple
            explode = [0.05, 0.05, 0.05]  # Slightly explode all slices

            # Only plot if there's data
            if sum(values) > 0:
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=labels,
                    colors=colors,
                    explode=explode,
                    autopct='%1.1f%%',
                    shadow=True,
                    startangle=90,
                    textprops={'fontsize': 12},
                )

                # Make percentage text bold
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                # Create title with year info if available
                title = 'Site Trends Over Time'
                if 'first_year' in trend_analysis and 'last_year' in trend_analysis:
                    first_yr = trend_analysis['first_year'].replace('Spill Events ', '')
                    last_yr = trend_analysis['last_year'].replace('Spill Events ', '')
                    title = f'Site Trends ({first_yr} → {last_yr})'
                
                ax.set_title(
                    title,
                    fontsize=16,
                    fontweight='bold',
                    pad=20
                )
            else:
                ax.text(
                    0.5, 0.5,
                    'No trend data available',
                    fontsize=14,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    color='gray',
                    bbox={
                        'boxstyle': 'round, pad=0.3',
                        'facecolor': 'lightgray',
                        'edgecolor': 'gray',
                        'linewidth': 1,
                    }
                )

            plt.tight_layout()

            if self.config.save_plots:
                fig.savefig(output_path, bbox_inches='tight', dpi=self.config.plot_dpi)
                logger.info(f"Trend visualization saved to {output_path}")

        return output_path

    def create_comprehensive_visualization(
        self,
        df: pd.DataFrame,
        company_stats: pd.DataFrame,
        regional_stats: pd.DataFrame,
        trend_analysis: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Create comprehensive multi-panel visualization.

        Like creating a **poster presentation for a science fair** -
        combines multiple charts into a big one poster showing all
        your findings at once!
        """
        logger.info("Creating comprehensive visualization...")

        # Ensure Avg_Annual_Spills exists
        if 'Avg_Annual_Spills' not in df.columns:
            spill_cols = [col for col in df.columns if 'Spill Events' in col]
            if spill_cols:
                df['Avg_Annual_Spills'] = df[spill_cols].mean(axis=1)
            else:
                df['Avg_Annual_Spills'] = 0

        with managed_figure(
            figsize=(self.config.figure_size[0] * 2, self.config.figure_size[1] * 2),
            dpi=self.config.plot_dpi
        ) as fig:
            # Create 2x2 grid of subplots
            axes = fig.subplots(2, 2)

            # Plot 1: Top Companies
            top_companies = company_stats.head(self.config.top_n_companies)
            sns.barplot(
                data=top_companies,
                y='Water company',
                x='avg_spills_per_site',
                ax=axes[0, 0],
                palette=self.config.palette
            )
            axes[0, 0].set_title('Top Companies by Average Annual Spills', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Average Annual Spills')
            axes[0, 0].set_ylabel('Water company')

            # Plot 2: Top Regions
            top_regions = regional_stats.head(self.config.top_n_regions)
            if not top_regions.empty:
                sns.barplot(
                    data=top_regions,
                    y='River Basin District',
                    x='avg_spills_per_site',
                    ax=axes[0, 1],
                    palette=self.config.palette
                )
                axes[0, 1].set_title('Top Regions by Average Annual Spills', fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel('Average Annual Spills')
                axes[0, 1].set_ylabel('River Basin District')
            else:
                axes[0, 1].text(0.5, 0.5, 'No regional data available', ha='center', va='center')
                axes[0, 1].set_title('Top Regions', fontsize=14, fontweight='bold')

            # Plot 3: Distribution
            cutoff = np.percentile(df['Avg_Annual_Spills'], self.config.outlier_percentile)
            plot_data = df[df['Avg_Annual_Spills'] <= cutoff]['Avg_Annual_Spills']

            axes[1, 0].hist(
                plot_data,
                bins=self.config.histogram_bins,
                alpha=0.7,
                edgecolor='black',
                color=sns.color_palette(self.config.palette)[0]
            )
            axes[1, 0].axvline(
                df['Avg_Annual_Spills'].mean(),
                color='red',
                linestyle='--',
                linewidth=2,
                label='Mean'
            )
            axes[1, 0].set_title('Distribution of Annual Spills', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Annual Spills')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Trends (if available)
            if trend_analysis['trend_available']:
                labels = ['Improving', 'Worsening', 'Stable']
                values = [
                    trend_analysis['improving_sites'],
                    trend_analysis['worsening_sites'],
                    trend_analysis['stable_sites']
                ]
                colors = ['#2ecc71', '#e74c3c', '#95a5a6']

                if sum(values) > 0:
                    axes[1, 1].pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                    axes[1, 1].set_title('Site Trends', fontsize=14, fontweight='bold')
                else:
                    axes[1, 1].text(
                        0.5, 0.5,
                        'No trend data',
                        ha='center',
                        va='center',
                        fontsize=12
                    )
                    axes[1, 1].set_title('Site Trends', fontsize=14, fontweight='bold')
                    axes[1, 1].axis('off')
            else:
                axes[1, 1].text(
                    0.5, 0.5,
                    'Trend data not available',
                    ha='center',
                    va='center',
                    fontsize=12
                )
                axes[1, 1].set_title('Site Trends', fontsize=14, fontweight='bold')
                axes[1, 1].axis('off')

            plt.tight_layout()

            if self.config.save_plots:
                fig.savefig(output_path, bbox_inches='tight', dpi=self.config.plot_dpi)
                logger.info(f"Comprehensive visualization saved to {output_path}")

        return output_path

    def generate_text_report(self, results: EDAResults) -> str:
        """
        Generate formatted text report of findings.

        Like writing a **book report** - summarizing all the important
        findings in a clear, easy-to-read format that anyone can understand.
        """
        report_lines = []
        report_lines.append('=' * 80)
        report_lines.append(f"{self.config.report_title}".center(80))
        report_lines.append(f"Generated: {results.timestamp}".center(80))
        report_lines.append("-" * 80)
        report_lines.append("")

        # Summary Metrics
        metrics = results.summary_metrics
        report_lines.append("📊 SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total sites analyzed: {metrics['total_sites']:,}")
        report_lines.append(f"Number of companies: {metrics['total_companies']}")
        report_lines.append(f"Number of regions: {metrics['total_regions']}")
        report_lines.append(f"Total annual spills: {metrics['spills']['total']:,}")
        report_lines.append(f"Average annual spills: {metrics['spills']['mean']:,.2f}")
        report_lines.append(f"Median annual spills: {metrics['spills']['median']:,.2f}")
        report_lines.append(f"Standard deviation: {metrics['spills']['std']:,.2f}")
        report_lines.append(f"Minimum annual spills: {metrics['spills']['min']:,.2f}")
        report_lines.append(f"Maximum annual spills: {metrics['spills']['max']:,.2f}")
        report_lines.append("")

        # Percentiles
        report_lines.append("📈 PERCENTILE BREAKDOWN")
        report_lines.append("-" * 80)
        report_lines.append(f"25th percentile: {metrics['spills']['percentile_25']:,.2f}")
        report_lines.append(f"75th percentile: {metrics['spills']['percentile_75']:,.2f}")
        report_lines.append(f"90th percentile: {metrics['spills']['percentile_90']:,.2f}")
        report_lines.append(f"95th percentile: {metrics['spills']['percentile_95']:,.2f}")
        report_lines.append(f"99th percentile: {metrics['spills']['percentile_99']:,.2f}")
        report_lines.append("")

        # Trend analysis
        if results.trend_analysis['trend_available']:
            report_lines.append("📉 TREND ANALYSIS")
            report_lines.append("-" * 80)
            trend = results.trend_analysis
            report_lines.append(f"Overall trend: {trend['overall_trend']:.4f}")
            report_lines.append(f"Sites improving: {trend['improving_sites']} "
                                f"({trend['improving_sites']/metrics['total_sites']*100:.1f}%)")
            report_lines.append(f"Sites worsening: {trend['worsening_sites']} "
                                f"({trend['worsening_sites']/metrics['total_sites']*100:.1f}%)")
            report_lines.append(f"Sites stable: {trend['stable_sites']} "
                                f"({trend['stable_sites']/metrics['total_sites']*100:.1f}%)")
            report_lines.append("")

        # Top Companies
        if results.company_stats is not None and not results.company_stats.empty:
            num_companies = min(5, len(results.company_stats))
            report_lines.append(f"🏢 TOP {num_companies} COMPANIES BY AVERAGE SPILLS")
            report_lines.append("-" * 80)
            for i, (idx, row) in enumerate(results.company_stats.head(5).iterrows()):
                report_lines.append(
                    f"{i+1}. {row['Water company']}: {row['avg_spills_per_site']:.2f} "
                    f"(Sites: {row['site_count']}, Total: {row['total_spills']:.0f})"
                )
            report_lines.append("")

        # Top Regions
        if results.regional_stats is not None and not results.regional_stats.empty:
            num_regions = min(5, len(results.regional_stats))
            report_lines.append(f"🌍 TOP {num_regions} REGIONS BY AVERAGE SPILLS")
            report_lines.append("-" * 80)
            for i, (idx, row) in enumerate(results.regional_stats.head(5).iterrows()):
                report_lines.append(
                    f"{i+1}. {row['River Basin District']}: {row['avg_spills_per_site']:.2f} "
                    f"(Sites: {row['site_count']}, Total: {row['total_spills']:.0f})"
                )
            report_lines.append("")

        # Warnings
        if results.warnings:
            report_lines.append("⚠️  WARNINGS")
            report_lines.append("-" * 80)
            for warning in results.warnings:
                report_lines.append(f"  • {warning}")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append(f"Processing time: {results.processing_time_seconds:.2f} seconds")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def perform_analysis(
        self,
        df: pd.DataFrame
    ) -> EDAResults:
        """
        Main function to perform comprehensive exploratory analysis.

        Returns:
            EDAResults object containing all analysis outputs

        Think of this as the **orchestra conductor** who coordinates
        all of the musicians (analysis methods) to play a beautiful
        symphony (comprehensive analysis) together!
        """
        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("Starting comprehensive exploratory data analysis")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info("=" * 60)

        try:
            # Step 1: Validate data
            self._validate_data(df)

            # Step 2: Calculate statistics
            logger.info("Phase 1/4: Calculating statistics...")
            company_stats = self.calculate_company_statistics(df)
            regional_stats = self.calculate_regional_statistics(df)
            trend_analysis = self.analyze_trends(df)
            summary_metrics = self.calculate_summary_metrics(df)

            # Step 3: Create visualizations
            logger.info("Phase 2/4: Creating visualizations...")
            plot_paths = []

            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Individual plots
            company_plot = self.create_company_visualization(
                company_stats,
                f"{self.config.output_dir}/company_analysis_{timestamp_str}.{self.config.plot_format}"
            )
            plot_paths.append(company_plot)

            regional_plot = self.create_regional_visualization(
                regional_stats,
                f"{self.config.output_dir}/regional_analysis_{timestamp_str}.{self.config.plot_format}"
            )
            plot_paths.append(regional_plot)

            distribution_plot = self.create_distribution_visualization(
                df,
                f"{self.config.output_dir}/distribution_analysis_{timestamp_str}.{self.config.plot_format}"
            )
            plot_paths.append(distribution_plot)

            trend_plot = self.create_trend_visualization(
                trend_analysis,
                f"{self.config.output_dir}/trend_analysis_{timestamp_str}.{self.config.plot_format}"
            )
            if trend_plot:
                plot_paths.append(trend_plot)

            # Comprehensive plot
            comprehensive_plot = self.create_comprehensive_visualization(
                df, company_stats, regional_stats, trend_analysis,
                f"{self.config.output_dir}/comprehensive_analysis_{timestamp_str}.{self.config.plot_format}"
            )
            plot_paths.append(comprehensive_plot)

            # Step 4: Create results object
            logger.info("Phase 3/4: Compiling results...")
            processing_time = (datetime.now() - start_time).total_seconds()

            results = EDAResults(
                timestamp=start_time.isoformat(),
                company_stats=company_stats,
                regional_stats=regional_stats,
                trend_analysis=trend_analysis,
                summary_metrics=summary_metrics,
                plot_paths=plot_paths,
                warnings=self.error_handler.warnings.copy(),
                processing_time_seconds=processing_time,
            )

            # Step 5: Generate reports
            logger.info("Phase 4/4: Generating reports...")

            # Text report
            text_report = self.generate_text_report(results)
            logger.info("\n" + text_report)

            # Save text report to file
            report_path = f"{self.config.output_dir}/eda_report_{timestamp_str}.txt"
            with open(report_path, 'w') as f:
                f.write(text_report)
            logger.info(f"Text report saved to {report_path}")

            # Save JSON results
            json_path = f"{self.config.output_dir}/eda_results_{timestamp_str}.json"
            results.save_to_json(json_path)

            logger.info("=" * 60)
            logger.info("Exploratory analysis completed successfully!")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"Results saved to: {self.config.output_dir}")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"Error during exploratory analysis: {str(e)}", exc_info=True)
            raise


# Convenience function for backward compatibility
def perform_exploratory_analysis(
    df: pd.DataFrame,
    config: Optional[EDAConfig] = None,
    **config_kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to perform EDA with default or custom configuration.

    Example usage:
        # Simple usage with defaults
        company_stats, regional_stats, trend = perform_exploratory_analysis(df)

        # With custom settings
        company_stats, regional_stats, trend = perform_exploratory_analysis(
            df,
            output_dir='my_analysis',
            top_n_companies=15,
            plot_dpi=150
        )

        # Using config object for full control
        config = EDAConfig(
            output_dir='production_analysis',
            save_plots=True,
            generate_html_report=True,
            top_n_companies=20
        )
        company_stats, regional_stats, trends = perform_exploratory_analysis(df, config=config)
    """
    if config is None:
        if config_kwargs:
            config = EDAConfig(**config_kwargs)
        else:
            config = EDAConfig()

    analyzer = WaterQualityEDA(config)
    results = analyzer.perform_analysis(df)

    # Return in backward-compatible format
    return (
        results.company_stats,
        results.regional_stats,
        results.trend_analysis
    )


# Example usage for production
if __name__ == "__main__":
    """
    Example demonstrating production usage patterns.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('eda_analysis.log'),
            logging.StreamHandler()
        ]
    )

    print("\n" + "=" * 80)
    print("WATER QUALITY EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Check if data file exists
    if DATA_FILE_PATH.exists():
        print(f"\nLoading data from: {DATA_FILE_PATH}")

        # Method 1: Simple usage with defaults - load from file
        print("\n" + "-" * 80)
        print("METHOD 1: Load and analyze real data")
        print("-" * 80)

        config = EDAConfig(
            filepath=str(DATA_FILE_PATH),
            output_dir='reports',
            top_n_companies=10,
            plot_dpi=150,
            save_plots=True
        )
        analyzer = WaterQualityEDA(config)

        # Load data using Dask
        df = analyzer.load_data()

        # Perform analysis
        results = analyzer.perform_analysis(df)

        print(f"\n✅ Analysis complete!")
        print(f"   - Companies analyzed: {len(results.company_stats)}")
        print(f"   - Regions analyzed: {len(results.regional_stats)}")
        print(f"   - Plots saved: {len(results.plot_paths)}")

    else:
        print(f"\n⚠️  Data file not found at: {DATA_FILE_PATH}")
        print("Creating sample data for demonstration...")

        # Create sample data for demonstration
        np.random.seed(42)
        df_sample = pd.DataFrame({
            'Avg_Annual_Spills': np.random.exponential(scale=10, size=1000),
            'Water Company': np.random.choice(
                ['Thames Water', 'Anglian Water', 'Severn Trent', 'United Utilities', 'Yorkshire Water'],
                1000
            ),
            'River Basin District': np.random.choice(
                ['Thames', 'Anglian', 'Severn', 'North West', 'Humber'],
                1000
            ),
            'Spill_Trend': np.random.normal(0, 5, 1000),
            'Latitude': np.random.uniform(50, 56, 1000),
            'Longitude': np.random.uniform(-5, 2, 1000)
        })

        # Method 2: Full control with config object
        print("\n" + "-" * 80)
        print("METHOD 2: Full control with results object")
        print("-" * 80)

        config = EDAConfig(
            output_dir='reports',
            top_n_companies=5,
            plot_dpi=150,
            save_plots=True
        )
        analyzer = WaterQualityEDA(config)
        results = analyzer.perform_analysis(df_sample)

        print(f"\n✅ Analysis complete!")
        print(f"   - Companies analyzed: {len(results.company_stats)}")
        print(f"   - Regions analyzed: {len(results.regional_stats)}")
        print(f"   - Processing time: {results.processing_time_seconds:.2f}s")

    print("\n" + "=" * 80)
    print("Analysis complete! Check the 'reports' directory for results.")
    print("=" * 80)
