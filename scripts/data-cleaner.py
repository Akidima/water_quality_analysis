"""
Water Data Cleaning Module
Purpose: Clean and preprocess water quality data with Pydantic validation and Dask parallel processing.
"""

import dask.dataframe as dd
import dask
import pandas as pd
import numpy as np
import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union, TYPE_CHECKING, cast
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

if TYPE_CHECKING:
    from pydantic_enhancements import ApplicationSettings, DataValidationError, DatasetStatistics
else:
    ApplicationSettings = Any  # type: ignore
    DataValidationError = Any  # type: ignore
    DatasetStatistics = Any  # type: ignore

# Import from data_loader module (handles hyphenated filename)
import importlib.util
import sys
data_loader_path = Path(__file__).parent / "data-loader.py"
spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
if spec is None:
    raise ImportError(f"Could not load module from {data_loader_path}")
if spec.loader is None:
    raise ImportError(f"Module spec has no loader for {data_loader_path}")
loader = spec.loader
data_loader = importlib.util.module_from_spec(spec)
sys.modules["data_loader"] = data_loader
loader.exec_module(data_loader)

DataConfig = data_loader.DataConfig
DataLoader = data_loader.DataLoader
ValidationReport = data_loader.ValidationReport
ExplorationMetadata = data_loader.ExplorationMetadata
ValidationMetadata = data_loader.ValidationMetadata
ValidationStats = data_loader.ValidationStats

# Configure module logger
logger = logging.getLogger(__name__)

# Import Pydantic enhancements
try:
    from pydantic_enhancements import (
        ColumnNameValidator, NumericStatistics, CategoricalStatistics,
        DatasetStatistics, ErrorHandler, ColumnValidationError,
        RangeValidationError, DataQualityError, DataValidationError,
        ApplicationSettings as _ApplicationSettings,
        load_settings
    )
    _ApplicationSettingsRuntime = _ApplicationSettings
except ImportError:
    logger.warning("Pydantic enhancements module not available. Using basic features only.")
    ColumnNameValidator = None
    ErrorHandler = None
    NumericStatistics = None
    CategoricalStatistics = None
    DatasetStatistics = None
    ColumnValidationError = Exception
    RangeValidationError = Exception
    DataQualityError = Exception
    DataValidationError = Exception
    _ApplicationSettingsRuntime = None
    load_settings = None


class DataCleanerConfig(BaseModel):
    """Configuration for data cleaning with Pydantic validation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Column Configuration
    required_columns: List[str] = Field(
        default_factory=lambda: ['Latitude', 'Longitude', 'Water company', 'River Basin District'],
        description="Columns that must be present in the dataset"
    )
    optional_columns: List[str] = Field(
        default_factory=lambda: ['Site Name', 'Receiving Environment', 'Permit Number'],
        description="Optional columns to use if present"
    )
    coordinate_columns: List[str] = Field(
        default_factory=lambda: ['Latitude', 'Longitude'],
        description="Geographic coordinate columns"
    )
    spill_year_columns: List[str] = Field(
        default_factory=lambda: [
            'Spill Events 2020', 'Spill Events 2021', 'Spill Events 2022', 
            'Spill Events 2023', 'Spill Events 2024', 'Spill Events 2025'
        ],
        description="Spill event columns by year"
    )
    text_columns: List[str] = Field(
        default_factory=lambda: ['Water company', 'River Basin District', 'Site Name', 'Receiving Environment'],
        description="Text columns to validate"
    )
    numeric_columns: List[str] = Field(
        default_factory=lambda: ['Latitude', 'Longitude'],
        description="Numeric columns"
    )

    # Validation Configuration
    lat_min: float = Field(default=-90.0, ge=-90.0, le=90.0)
    lat_max: float = Field(default=90.0, ge=-90.0, le=90.0)
    lon_min: float = Field(default=-180.0, ge=-180.0, le=180.0)
    lon_max: float = Field(default=180.0, ge=-180.0, le=180.0)
    spill_year_min: int = Field(default=2020, ge=2000, le=2030)
    spill_year_max: int = Field(default=2025, ge=2000, le=2030)
    
    # Thresholds
    strict_validation: bool = Field(default=True)
    missing_value_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    duplicate_value_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    # Cleaning parameters
    remove_duplicates: bool = Field(default=True)
    fill_missing_values: bool = Field(default=False)
    fill_value: Optional[Union[str, int, float]] = Field(default=None)
    remove_outliers: bool = Field(default=True)
    outlier_std_threshold: float = Field(default=3.0, gt=0.0)
    min_valid_spill_years: int = Field(default=3, ge=1)

    # Behavior flags
    strict_mode: bool = Field(default=True)
    remove_invalid_coordinates: bool = Field(default=True)
    remove_invalid_spill_years: bool = Field(default=True)
    remove_invalid_text_values: bool = Field(default=True)
    create_backup: bool = Field(default=True)

    # Output configuration
    save_cleaning_report: bool = Field(default=True)
    output_directory: str = Field(default='cleaned_data')
    output_filename: str = Field(default='cleaned_data.csv')
    report_output_path: str = Field(default='cleaning_report.json')

    # Logging configuration
    log_level: str = Field(default='INFO')
    log_format: str = Field(default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Dask configuration
    n_partitions: Optional[int] = Field(default=None, gt=0)
    
    @field_validator('lat_min', 'lat_max')
    @classmethod
    def validate_latitude_range(cls, v: float) -> float:
        """Validate latitude values."""
        if not -90.0 <= v <= 90.0:
            raise ValueError(f"Latitude must be between -90 and 90, got {v}")
        return v
    
    @field_validator('lon_min', 'lon_max')
    @classmethod
    def validate_longitude_range(cls, v: float) -> float:
        """Validate longitude values."""
        if not -180.0 <= v <= 180.0:
            raise ValueError(f"Longitude must be between -180 and 180, got {v}")
        return v
    
    @model_validator(mode='after')
    def validate_min_max_ranges(self):
        """Validate min < max."""
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min must be < lat_max")
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min must be < lon_max")
        if self.spill_year_min >= self.spill_year_max:
            raise ValueError(f"spill_year_min must be < spill_year_max")
        return self


class CleaningReport(BaseModel):
    """Pydantic model for comprehensive cleaning report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    start_time: str
    end_time: Optional[str] = None
    duration: Optional[float] = Field(default=None, ge=0.0)
    config: Dict[str, Any]
    validation: Optional[ValidationReport] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    original_shape: Tuple[int, int]
    original_columns: List[str]
    original_dtypes: Dict[str, str]
    original_missing_values: Dict[str, int]
    original_duplicates: int = Field(ge=0)
    
    cleaned_shape: Tuple[int, int]
    cleaned_columns: List[str]
    cleaned_dtypes: Dict[str, str]
    cleaned_missing_values: Dict[str, int]
    cleaned_duplicates: int = Field(ge=0)
    
    removal_breakdown: Dict[str, int] = Field(default_factory=dict)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    processing_time_seconds: float = Field(default=0.0, ge=0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'config': self.config,
            'validation': self.validation.model_dump() if self.validation else None,
            'errors': self.errors,
            'warnings': self.warnings,
            'original_shape': self.original_shape,
            'original_columns': self.original_columns,
            'original_dtypes': self.original_dtypes,
            'original_missing_values': self.original_missing_values,
            'original_duplicates': self.original_duplicates,
            'cleaned_shape': self.cleaned_shape,
            'cleaned_columns': self.cleaned_columns,
            'cleaned_dtypes': self.cleaned_dtypes,
            'cleaned_missing_values': self.cleaned_missing_values,
            'cleaned_duplicates': self.cleaned_duplicates,
            'removal_breakdown': self.removal_breakdown,
            'quality_metrics': self.quality_metrics,
            'processing_time_seconds': self.processing_time_seconds
        }

    def save_to_file(self, filepath: str) -> None:
        """Save report to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=4, default=str)
            logger.info(f"Report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
            raise


class WaterDataCleaner:
    """
    Production-ready water quality data cleaner with Dask parallel processing.
    
    Performs comprehensive data cleaning operations using parallel processing.
    """

    def __init__(self, config: Optional[DataCleanerConfig] = None, settings: Optional["ApplicationSettings"] = None):
        """Initialize cleaner with configuration."""
        self.config = config or DataCleanerConfig()
        self.settings = settings
        self.report: Optional[CleaningReport] = None
        self.cleaned_df: Optional[dd.DataFrame] = None
        self.error_handler = ErrorHandler() if ErrorHandler else None
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format
        )
    
    def _create_backup(self, df: dd.DataFrame, output_dir: str) -> None:
        """Create backup of original data."""
        if not self.config.create_backup:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"backup_{timestamp}.csv"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            backup_path = os.path.join(output_dir, backup_filename)
            df.to_csv(backup_path, index=False, single_file=True)
            logger.info(f"Backup created at {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            raise

    def _validate_column_names(self, df: dd.DataFrame) -> None:
        """Validate column names against standard patterns (Enhancement 1)."""
        if ColumnNameValidator is None:
            return
        
        try:
            columns = df.columns.tolist()
            results = ColumnNameValidator.validate_columns(columns)
            
            # Log warnings for non-standard columns
            for warning in results.get('warnings', []):
                msg = f"Non-standard column name: '{warning}'"
                if self.error_handler:
                    self.error_handler.add_warning(msg)
                logger.warning(msg)
            
            # Log suggestions for corrections
            suggestions_raw = results.get('suggestions', [])
            suggestions_list: List[Dict[str, Any]] = cast(List[Dict[str, Any]], suggestions_raw)
            for suggestion_dict in suggestions_list:
                if isinstance(suggestion_dict, dict):
                    col = str(suggestion_dict.get('column', ''))
                    suggestions = list(suggestion_dict.get('suggestions', []))
                    msg = f"Column '{col}' may have typo. Suggestions: {suggestions}"
                    if self.error_handler:
                        self.error_handler.add_warning(msg)
                    logger.info(msg)
                
        except Exception as e:
            logger.warning(f"Column name validation failed: {e}")
    
    def _validate_columns(self, df: dd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Validate required and optional columns.
        
        Returns:
            Tuple of (available_required, missing_required, available_optional, missing_optional)
        """
        # Enhancement 1: Validate column naming patterns
        self._validate_column_names(df)
        
        available_required = [col for col in self.config.required_columns if col in df.columns]
        missing_required = list(set(self.config.required_columns) - set(df.columns))
        available_optional = [col for col in self.config.optional_columns if col in df.columns]
        missing_optional = list(set(self.config.optional_columns) - set(df.columns))
        
        if missing_required:
            msg = f"Missing required columns: {missing_required}"
            logger.warning(msg)
            if self.error_handler and ColumnValidationError is not Exception:
                for col in missing_required:
                    error_instance = ColumnValidationError(col, "Required column is missing")
                    # When ColumnValidationError is not Exception, it's a subclass of DataValidationError
                    # Runtime check ensures type safety, but type checker needs help
                    self.error_handler.add_error(error_instance)  # type: ignore[arg-type]
        
        if missing_optional:
            logger.info(f"Missing optional columns: {missing_optional}")
            
        return available_required, missing_required, available_optional, missing_optional

    def _clean_coordinates(self, df: dd.DataFrame) -> dd.DataFrame:
        """Clean and validate geographic coordinates using Dask (with Enhancement 3)."""
        initial_len = len(df)
        coord_cols = [col for col in df.columns if col in self.config.coordinate_columns]
        
        if not coord_cols:
            logger.warning("No coordinate columns found. Skipping.")
            return df
        
        # Remove rows with missing coordinates
        df = df.dropna(subset=coord_cols)
        missing_coords_removed = initial_len - len(df)
        
        if missing_coords_removed > 0:
            logger.warning(f"Removed {missing_coords_removed} rows with missing coordinates")
            if self.report:
                self.report.removal_breakdown['missing_coordinates'] = missing_coords_removed
            
            # Enhancement 3: Use custom error for data quality issues
            if self.error_handler and DataQualityError is not Exception:
                error_instance = DataQualityError(
                    "Missing geographic coordinates",
                    missing_coords_removed,
                    initial_len
                )
                self.error_handler.add_error(error_instance)  # type: ignore[arg-type]
        
        # Validate latitude
        if 'Latitude' in df.columns:
            valid_lat_mask = (df['Latitude'] >= self.config.lat_min) & (df['Latitude'] <= self.config.lat_max)
            invalid_lat_count = (~valid_lat_mask).sum().compute()
            
            if invalid_lat_count > 0:
                logger.warning(f"Removing {invalid_lat_count} rows with invalid latitude")
                df = df[valid_lat_mask]
                if self.report:
                    self.report.removal_breakdown['invalid_latitude'] = int(invalid_lat_count)
                
                # Enhancement 3: Use custom range validation error
                if self.error_handler and RangeValidationError is not Exception:
                    error_instance = RangeValidationError(
                        'Latitude',
                        f"{invalid_lat_count} values",
                        self.config.lat_min,
                        self.config.lat_max
                    )
                    self.error_handler.add_error(error_instance)  # type: ignore[arg-type]
        
        # Validate longitude
        if 'Longitude' in df.columns:
            valid_lon_mask = (df['Longitude'] >= self.config.lon_min) & (df['Longitude'] <= self.config.lon_max)
            invalid_lon_count = (~valid_lon_mask).sum().compute()
            
            if invalid_lon_count > 0:
                logger.warning(f"Removing {invalid_lon_count} rows with invalid longitude")
                df = df[valid_lon_mask]
                if self.report:
                    self.report.removal_breakdown['invalid_longitude'] = int(invalid_lon_count)
                
                # Enhancement 3: Use custom range validation error
                if self.error_handler and RangeValidationError is not Exception:
                    error_instance = RangeValidationError(
                        'Longitude',
                        f"{invalid_lon_count} values",
                        self.config.lon_min,
                        self.config.lon_max
                    )
                    self.error_handler.add_error(error_instance)  # type: ignore[arg-type]
        
        return df

    def _clean_spill_events(self, df: dd.DataFrame) -> dd.DataFrame:
        """Clean and validate spill events with outlier detection using Dask."""
        existing_spill_cols = [col for col in df.columns if col in self.config.spill_year_columns]
        
        if not existing_spill_cols:
            logger.warning("No spill event columns found. Skipping.")
            return df
        
        logger.info(f"Cleaning {len(existing_spill_cols)} spill event columns")
        
        # Convert to numeric
        for col in existing_spill_cols:
            df[col] = dd.to_numeric(df[col], errors='coerce')
        
        if self.config.remove_outliers:
            for col in existing_spill_cols:
                col_mean = df[col].mean().compute()
                col_std = df[col].std().compute()
                
                if col_std > 0:
                    z_scores = (df[col] - col_mean) / col_std
                    outlier_mask = abs(z_scores) > self.config.outlier_std_threshold
                    outlier_count = outlier_mask.sum().compute()
                    
                    if outlier_count > 0:
                        logger.warning(f"Removing {outlier_count} outliers from {col}")
                        df[col] = df[col].where(~outlier_mask, np.nan)
        
        # Remove rows with insufficient valid spill years
        if self.config.remove_invalid_spill_years:
            valid_counts = (~df[existing_spill_cols].isnull()).sum(axis=1)
            invalid_mask = valid_counts < self.config.min_valid_spill_years
            invalid_count = invalid_mask.sum().compute()
            
            if invalid_count > 0:
                logger.warning(f"Removing {invalid_count} rows with < {self.config.min_valid_spill_years} valid spill years")
                df = df[~invalid_mask]
                if self.report:
                    self.report.removal_breakdown['insufficient_spill_years'] = int(invalid_count)
        
        return df

    def _clean_text_fields(self, df: dd.DataFrame) -> dd.DataFrame:
        """Clean and validate text fields using Dask."""
        existing_text_cols = [col for col in df.columns if col in self.config.text_columns]
        
        if not existing_text_cols:
            logger.info("No text columns found. Skipping.")
            return df
        
        for col in existing_text_cols:
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('', np.nan)
        
        if self.config.remove_invalid_text_values:
            text_valid_mask = (~df[existing_text_cols].isnull()).any(axis=1)
            invalid_count = (~text_valid_mask).sum().compute()
            
            if invalid_count > 0:
                logger.warning(f"Removing {invalid_count} rows with all text fields empty")
                df = df[text_valid_mask]
                if self.report:
                    self.report.removal_breakdown['empty_text_fields'] = int(invalid_count)
        
        return df

    def _remove_duplicates(self, df: dd.DataFrame) -> dd.DataFrame:
        """Remove duplicate rows using Dask."""
        if not self.config.remove_duplicates:
            return df
        
        initial_len = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_len - len(df)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
            if self.report:
                self.report.removal_breakdown['duplicates'] = duplicates_removed
        
        return df

    def _handle_missing_values(self, df: dd.DataFrame) -> dd.DataFrame:
        """Handle missing values according to configuration using Dask."""
        if not self.config.fill_missing_values:
            missing_pct = df.isnull().sum(axis=1) / len(df.columns)
            rows_to_remove = missing_pct > self.config.missing_value_threshold
            rows_to_remove_count = rows_to_remove.sum().compute()
            
            if rows_to_remove_count > 0:
                logger.warning(f"Removing {rows_to_remove_count} rows exceeding missing value threshold")
                df = df[~rows_to_remove]
                if self.report:
                    self.report.removal_breakdown['high_missing_values'] = int(rows_to_remove_count)
        else:
            if self.config.fill_value is not None:
                df = df.fillna(self.config.fill_value)
                logger.info(f"Filled missing values with: {self.config.fill_value}")
        
        return df
    
    def _generate_statistics(self, df: dd.DataFrame) -> Optional[Any]:  # type: ignore[assignment]
        """Generate comprehensive statistics using Pydantic models (Enhancement 2) - Optimized."""
        if not DatasetStatistics or not NumericStatistics or not CategoricalStatistics:
            return None
        
        try:
            logger.info("Generating statistical summaries...")
            
            df_len = len(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            # Optimization: Convert to pandas if dataset is small enough (faster for small data)
            # Threshold: 50k rows or less
            use_pandas = df_len <= 50000
            
            if use_pandas:
                logger.info(f"Dataset size ({df_len} rows) is small enough, using pandas for faster statistics")
                df_pd = df.compute()
            else:
                df_pd = None
            
            numeric_stats = {}
            categorical_stats = {}
            
            # Generate numeric statistics - batch compute all stats at once
            if numeric_cols:
                if use_pandas and df_pd is not None:
                    # Use pandas describe() for fast computation
                    assert df_pd is not None  # Type narrowing for type checker
                    desc = df_pd[numeric_cols].describe()
                    counts = df_pd[numeric_cols].count()
                    missing_counts = df_len - counts
                    
                    for col in numeric_cols:
                        try:
                            count = int(counts[col])
                            missing = int(missing_counts[col])
                            missing_pct = (missing / df_len * 100) if df_len > 0 else 0
                            
                            if count > 0 and NumericStatistics is not None:
                                numeric_stats[col] = NumericStatistics(
                                    column_name=col,
                                    count=count,
                                    mean=float(desc.loc['mean', col]),
                                    std=float(desc.loc['std', col]),
                                    min=float(desc.loc['min', col]),
                                    median=float(desc.loc['50%', col]),
                                    q25=float(desc.loc['25%', col]),
                                    q75=float(desc.loc['75%', col]),
                                    max=float(desc.loc['max', col]),
                                    missing_count=missing,
                                    missing_percent=float(missing_pct)
                                )
                        except Exception as e:
                            logger.warning(f"Failed to generate statistics for {col}: {e}")
                else:
                    # Batch compute all numeric statistics using dask.compute()
                    # Process columns in batches to avoid memory issues
                    logger.info(f"Batch computing statistics for {len(numeric_cols)} numeric columns...")
                    
                    batch_size = 5  # Process 5 columns at a time
                    for batch_start in range(0, len(numeric_cols), batch_size):
                        batch_cols = numeric_cols[batch_start:batch_start + batch_size]
                        logger.info(f"Processing columns {batch_start + 1}-{min(batch_start + batch_size, len(numeric_cols))} of {len(numeric_cols)}...")
                        
                        # Prepare all computations for this batch
                        all_tasks = []
                        task_map = {}  # Map to track which task belongs to which column/stat
                        
                        for col in batch_cols:
                            col_data = df[col]
                            tasks = [
                                col_data.count(),
                                col_data.mean(),
                                col_data.std(),
                                col_data.min(),
                                col_data.quantile(0.25),
                                col_data.quantile(0.5),
                                col_data.quantile(0.75),
                                col_data.max()
                            ]
                            all_tasks.extend(tasks)
                            task_map[col] = len(all_tasks) - len(tasks)
                        
                        # Compute all stats for this batch in one go
                        try:
                            results = dask.compute(*all_tasks)
                            
                            # Process results for each column in this batch
                            for col in batch_cols:
                                try:
                                    idx = task_map[col]
                                    count, mean, std, min_val, q25, median, q75, max_val = results[idx:idx+8]
                                    
                                    count = int(count)
                                    missing = df_len - count
                                    missing_pct = (missing / df_len * 100) if df_len > 0 else 0
                                    
                                    if count > 0 and NumericStatistics is not None:
                                        numeric_stats[col] = NumericStatistics(
                                            column_name=col,
                                            count=count,
                                            mean=float(mean),
                                            std=float(std),
                                            min=float(min_val),
                                            median=float(median),
                                            q25=float(q25),
                                            q75=float(q75),
                                            max=float(max_val),
                                            missing_count=int(missing),
                                            missing_percent=float(missing_pct)
                                        )
                                except Exception as e:
                                    logger.warning(f"Failed to generate statistics for {col}: {e}")
                        except Exception as e:
                            logger.warning(f"Failed to compute batch for columns {batch_cols}: {e}")
            
            # Generate categorical statistics - batch compute
            if categorical_cols:
                if use_pandas and df_pd is not None:
                    # Use pandas for fast computation
                    assert df_pd is not None  # Type narrowing for type checker
                    for col in categorical_cols:
                        try:
                            col_data = df_pd[col]
                            count = int(col_data.count())
                            missing = df_len - count
                            missing_pct = (missing / df_len * 100) if df_len > 0 else 0
                            
                            if count > 0 and CategoricalStatistics is not None:
                                nunique = int(col_data.nunique())
                                mode_val = col_data.mode()
                                top_value = mode_val.iloc[0] if len(mode_val) > 0 else None
                                top_freq = int((col_data == top_value).sum()) if top_value else 0
                                
                                categorical_stats[col] = CategoricalStatistics(
                                    column_name=col,
                                    count=count,
                                    unique_count=nunique,
                                    unique_percent=float(nunique / count * 100) if count > 0 else 0,
                                    top_value=str(top_value) if top_value else None,
                                    top_frequency=top_freq,
                                    missing_count=int(missing),
                                    missing_percent=float(missing_pct)
                                )
                        except Exception as e:
                            logger.warning(f"Failed to generate statistics for {col}: {e}")
                else:
                    # Batch compute categorical statistics
                    logger.info(f"Batch computing statistics for {len(categorical_cols)} categorical columns...")
                    
                    batch_size = 10  # Process 10 categorical columns at a time (lighter operations)
                    for batch_start in range(0, len(categorical_cols), batch_size):
                        batch_cols = categorical_cols[batch_start:batch_start + batch_size]
                        logger.info(f"Processing columns {batch_start + 1}-{min(batch_start + batch_size, len(categorical_cols))} of {len(categorical_cols)}...")
                        
                        # Prepare all computations for this batch
                        all_tasks = []
                        task_map = {}
                        
                        for col in batch_cols:
                            col_data = df[col]
                            tasks = [
                                col_data.count(),
                                col_data.nunique(),
                                col_data.mode()
                            ]
                            all_tasks.extend(tasks)
                            task_map[col] = {
                                'idx': len(all_tasks) - len(tasks),
                                'col_data': col_data
                            }
                        
                        # Compute count, nunique, and mode for all columns in batch
                        try:
                            results = dask.compute(*all_tasks)
                            
                            # Process results and prepare frequency computations
                            freq_computations = []
                            
                            for col in batch_cols:
                                try:
                                    idx = task_map[col]['idx']
                                    count, nunique, mode_val = results[idx:idx+3]
                                    
                                    count = int(count)
                                    missing = df_len - count
                                    missing_pct = (missing / df_len * 100) if df_len > 0 else 0
                                    
                                    if count > 0 and CategoricalStatistics is not None:
                                        nunique = int(nunique)
                                        top_value = mode_val.iloc[0] if len(mode_val) > 0 else None
                                        
                                        # Prepare frequency computation if needed
                                        if top_value:
                                            freq_computations.append(
                                                (task_map[col]['col_data'] == top_value).sum()
                                            )
                                        
                                        # Store intermediate results
                                        task_map[col]['stats'] = {
                                            'count': count,
                                            'nunique': nunique,
                                            'top_value': top_value,
                                            'missing': missing,
                                            'missing_pct': missing_pct,
                                            'freq_idx': len(freq_computations) - 1 if top_value else None
                                        }
                                except Exception as e:
                                    logger.warning(f"Failed to process results for {col}: {e}")
                            
                            # Compute frequencies in batch
                            if freq_computations:
                                freq_results = dask.compute(*freq_computations)
                                
                                # Create stats for all columns in batch
                                for col in batch_cols:
                                    if 'stats' in task_map[col]:
                                        stats = task_map[col]['stats']
                                        
                                        # Get frequency if top_value exists
                                        if stats['freq_idx'] is not None:
                                            top_freq = int(freq_results[stats['freq_idx']])
                                        else:
                                            top_freq = 0
                                        
                                        if CategoricalStatistics is not None:
                                            categorical_stats[col] = CategoricalStatistics(
                                                column_name=col,
                                                count=stats['count'],
                                                unique_count=stats['nunique'],
                                                unique_percent=float(stats['nunique'] / stats['count'] * 100) if stats['count'] > 0 else 0,
                                                top_value=str(stats['top_value']) if stats['top_value'] else None,
                                                top_frequency=top_freq,
                                                missing_count=int(stats['missing']),
                                                missing_percent=float(stats['missing_pct'])
                                            )
                            else:
                                # No frequencies to compute, create stats without top_freq
                                for col in batch_cols:
                                    if 'stats' in task_map[col]:
                                        stats = task_map[col]['stats']
                                        if CategoricalStatistics is not None:
                                            categorical_stats[col] = CategoricalStatistics(
                                                column_name=col,
                                                count=stats['count'],
                                                unique_count=stats['nunique'],
                                                unique_percent=float(stats['nunique'] / stats['count'] * 100) if stats['count'] > 0 else 0,
                                                top_value=str(stats['top_value']) if stats['top_value'] else None,
                                                top_frequency=0,
                                                missing_count=int(stats['missing']),
                                                missing_percent=float(stats['missing_pct'])
                                            )
                        except Exception as e:
                            logger.warning(f"Failed to compute batch for categorical columns {batch_cols}: {e}")
            
            # Calculate overall metrics - batch compute
            logger.info("Computing overall metrics...")
            mem_usage, null_sum, dup_count = dask.compute(
                df.memory_usage(deep=True).sum() / 1e6,
                df.isnull().sum().sum(),
                df.map_partitions(lambda x: x.duplicated().sum(), meta=('x', 'i8')).sum()
            )
            
            overall_missing_pct = (float(null_sum) / (df_len * len(df.columns)) * 100) if df_len > 0 else 0
            
            if DatasetStatistics is None:
                return None
            
            stats = DatasetStatistics(
                total_rows=df_len,
                total_columns=len(df.columns),
                numeric_columns=numeric_stats,
                categorical_columns=categorical_stats,
                memory_usage_mb=float(mem_usage),
                overall_missing_percent=float(overall_missing_pct),
                duplicate_rows=int(dup_count)
            )
            
            logger.info(f"Overall data quality score: {stats.overall_quality_score():.2f}/100")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            return None

    def clean_data(
        self,
        df: Union[dd.DataFrame, pd.DataFrame],
        output_dir: Optional[str] = None
    ) -> Tuple[dd.DataFrame, CleaningReport]:
        """
        Main cleaning pipeline using Dask for parallel processing.
        
        Args:
            df: Input DataFrame (Dask or Pandas)
            output_dir: Directory to save outputs
            
        Returns:
            Tuple of (cleaned_df, cleaning_report)
        """
        start_time = datetime.now()
        logger.info("Starting data cleaning pipeline")
        
        # Convert pandas to dask if needed
        if isinstance(df, pd.DataFrame):
            logger.info("Converting Pandas DataFrame to Dask DataFrame")
            n_partitions = self.config.n_partitions or max(1, len(df) // 10000)
            df = dd.from_pandas(df, npartitions=n_partitions)
        
        output_dir = output_dir or self.config.output_directory
        # At this point, df is guaranteed to be a dd.DataFrame after conversion above
        self._create_backup(cast(dd.DataFrame, df), output_dir)
        
        # Collect initial statistics
        logger.info("Collecting initial statistics")
        initial_len = len(df)
        initial_cols = df.columns.tolist()
        initial_dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        initial_missing = df.isnull().sum().compute().to_dict()
        initial_missing_int = {k: int(v) for k, v in initial_missing.items()}
        
        try:
            initial_dups = df.map_partitions(lambda x: x.duplicated().sum(), meta=('x', 'i8')).sum().compute()
        except Exception:
            initial_dups = 0
            logger.warning("Could not calculate initial duplicates")
        
        # Initialize report
        self.report = CleaningReport(
            start_time=start_time.isoformat(),
            config=self.config.model_dump(),
            original_shape=(initial_len, len(initial_cols)),
            original_columns=initial_cols,
            original_dtypes=initial_dtypes,
            original_missing_values=initial_missing_int,
            original_duplicates=int(initial_dups),
            cleaned_shape=(0, 0),
            cleaned_columns=[],
            cleaned_dtypes={},
            cleaned_missing_values={},
            cleaned_duplicates=0
        )
        
        try:
            # Cleaning steps
            # At this point, df is guaranteed to be a dd.DataFrame after conversion above
            df_dask = cast(dd.DataFrame, df)
            
            logger.info("Step 1: Validating columns")
            avail_req, miss_req, _, _ = self._validate_columns(df_dask)
            
            if miss_req and self.config.strict_mode:
                error_msg = f"Missing required columns: {miss_req}"
                self.report.errors.append(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Step 2: Cleaning coordinates")
            df = self._clean_coordinates(df_dask)
            
            logger.info("Step 3: Cleaning spill events")
            df = self._clean_spill_events(df)
            
            logger.info("Step 4: Cleaning text fields")
            df = self._clean_text_fields(df)
            
            logger.info("Step 5: Removing duplicates")
            df = self._remove_duplicates(df)
            
            logger.info("Step 6: Handling missing values")
            df = self._handle_missing_values(df)
            
            # Collect final statistics
            logger.info("Collecting final statistics")
            final_len = len(df)
            final_cols = df.columns.tolist()
            final_dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
            final_missing = df.isnull().sum().compute().to_dict()
            final_missing_int = {k: int(v) for k, v in final_missing.items()}
            
            try:
                final_dups = df.map_partitions(lambda x: x.duplicated().sum(), meta=('x', 'i8')).sum().compute()
            except Exception:
                final_dups = 0
            
            # Update report
            self.report.cleaned_shape = (final_len, len(final_cols))
            self.report.cleaned_columns = final_cols
            self.report.cleaned_dtypes = final_dtypes
            self.report.cleaned_missing_values = final_missing_int
            self.report.cleaned_duplicates = int(final_dups)
            
            rows_removed = initial_len - final_len
            rows_retained_pct = (final_len / initial_len * 100) if initial_len > 0 else 0
            
            self.report.quality_metrics = {
                'rows_removed': rows_removed,
                'rows_retained_percent': rows_retained_pct,
                'initial_missing_percent': (sum(initial_missing.values()) / (initial_len * len(initial_cols)) * 100) if initial_len > 0 else 0,
                'final_missing_percent': (sum(final_missing.values()) / (final_len * len(final_cols)) * 100) if final_len > 0 else 0,
                'duplicate_reduction': initial_dups - final_dups
            }
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            self.report.end_time = end_time.isoformat()
            self.report.duration = processing_time
            self.report.processing_time_seconds = processing_time
            
            logger.info(
                f"Cleaning completed: {initial_len} -> {final_len} rows "
                f"({rows_retained_pct:.1f}% retained) in {processing_time:.2f}s"
            )
            
            # Enhancement 2: Generate statistical summaries
            dataset_stats = self._generate_statistics(df)
            if dataset_stats:
                summary = dataset_stats.summary_report()
                self.report.quality_metrics.update({
                    'overall_quality_score': summary['quality_score'],
                    'column_quality_scores': summary['column_quality_scores']
                })
                logger.info(f"Overall data quality score: {summary['quality_score']:.2f}/100")
            
            # Enhancement 3: Add error handler report to cleaning report
            if self.error_handler:
                error_report = self.error_handler.get_error_report()
                if error_report['errors']:
                    self.report.errors.extend([err['message'] for err in error_report['errors']])
                if error_report['warnings']:
                    self.report.warnings.extend(error_report['warnings'])
            
            # Save outputs
            if self.config.save_cleaning_report:
                output_path = os.path.join(output_dir, self.config.output_filename)
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Saving cleaned data to {output_path}")
                df.to_csv(output_path, index=False, single_file=True)
                
                report_path = os.path.join(output_dir, self.config.report_output_path)
                self.report.save_to_file(report_path)
            
            self.cleaned_df = df
            return df, self.report
            
        except Exception as e:
            logger.error(f"Error during cleaning: {str(e)}", exc_info=True)
            if self.report:
                self.report.errors.append(str(e))
                self.report.end_time = datetime.now().isoformat()
            raise


# Convenience functions

def clean_water_data_with_settings(
    filepath: str,
    settings: Optional["ApplicationSettings"] = None,
    env_file: Optional[Path] = None
) -> Tuple[dd.DataFrame, CleaningReport]:
    """
    Clean water data using ApplicationSettings (Enhancement 4).
    
    Args:
        filepath: Path to the CSV file
        settings: Application settings instance (or None to load from environment)
        env_file: Optional path to .env file
        
    Returns:
        Tuple of (cleaned_df, cleaning_report)
    """
    # Load settings from environment or .env file
    if settings is None:
        settings = load_settings(env_file) if load_settings else (_ApplicationSettingsRuntime() if _ApplicationSettingsRuntime else None)
    
    # Configure logging
    if settings is not None:
        if hasattr(settings, 'configure_logging'):
            settings.configure_logging()  # type: ignore[union-attr]
        logger.info(f"Using {settings.app_name} v{settings.app_version}")  # type: ignore[union-attr]
    
    logger.info(f"Loading data from {filepath}")
    
    # Load data using DataLoader with settings
    if settings is not None and hasattr(settings, 'to_loader_config'):
        loader_config_dict = settings.to_loader_config()  # type: ignore[misc]
        loader_config_dict['filepath'] = filepath
        data_config = DataConfig(**loader_config_dict)
    else:
        data_config = DataConfig(filepath=filepath)
    
    loader = DataLoader(data_config)
    df, load_report = loader.load_and_explore_data()
    logger.info("Data loaded successfully")
    
    # Create cleaner configuration from settings
    if settings is not None and hasattr(settings, 'to_cleaner_config'):
        cleaner_config_dict = settings.to_cleaner_config()  # type: ignore[misc]
        cleaner_config = DataCleanerConfig(**cleaner_config_dict)
    else:
        cleaner_config = DataCleanerConfig()
    
    # Clean the data
    cleaner = WaterDataCleaner(cleaner_config, settings=settings)
    if settings is not None:
        cleaned_df, cleaning_report = cleaner.clean_data(df, str(settings.export_directory))  # type: ignore[misc]
    else:
        cleaned_df, cleaning_report = cleaner.clean_data(df)
    
    return cleaned_df, cleaning_report


def clean_water_data(
    filepath: str,
    config: Optional[DataCleanerConfig] = None,
    output_dir: Optional[str] = None,
    use_data_loader: bool = True
) -> Tuple[dd.DataFrame, CleaningReport]:
    """
    Convenience function to load and clean water quality data.
    
    Args:
        filepath: Path to the CSV file
        config: Data cleaner configuration
        output_dir: Output directory for cleaned data
        use_data_loader: Whether to use DataLoader for loading (recommended)
        
    Returns:
        Tuple of (cleaned_df, cleaning_report)
    """
    logger.info(f"Loading data from {filepath}")
    
    if use_data_loader:
        # Use DataLoader for optimized loading
        data_config = DataConfig(filepath=filepath)
        loader = DataLoader(data_config)
        df, load_report = loader.load_and_explore_data()
        logger.info("Data loaded successfully with DataLoader")
    else:
        # Direct Dask loading
        df = dd.read_csv(filepath)
        logger.info("Data loaded successfully with Dask")
    
    # Clean the data
    cleaner = WaterDataCleaner(config)
    cleaned_df, cleaning_report = cleaner.clean_data(df, output_dir)
    
    return cleaned_df, cleaning_report


def validate_config(config: DataCleanerConfig) -> bool:
    """
    Validate a DataCleanerConfig instance.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    try:
        # Pydantic validation happens automatically
        config.model_validate(config.model_dump())
        logger.info("Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define paths
    data_path = Path(__file__).parent.parent / 'data' / 'national_water_plan.csv'
    output_dir = Path(__file__).parent.parent / 'export' / 'cleaned_data'
    
    print("="*80)
    print("WATER DATA CLEANER - Production Version with Pydantic & Dask")
    print("="*80)
    print(f"\nInput: {data_path}")
    print(f"Output: {output_dir}\n")
    
    try:
        # Create custom configuration
        config = DataCleanerConfig(
            strict_mode=True,
            remove_duplicates=True,
            remove_outliers=True,
            outlier_std_threshold=3.0,
            min_valid_spill_years=3,
            save_cleaning_report=True,
            output_directory=str(output_dir),
            output_filename='cleaned_water_data.csv',
            report_output_path='cleaning_report.json',
            n_partitions=4  # Adjust based on your system
        )
        
        # Validate configuration
        print("Validating configuration...")
        validate_config(config)
        print("✓ Configuration valid\n")
        
        # Clean the data
        print("Starting data cleaning pipeline...\n")
        cleaned_df, report = clean_water_data(
            filepath=str(data_path),
            config=config,
            output_dir=str(output_dir),
            use_data_loader=True
        )
        
        # Print summary
        print("\n" + "="*80)
        print("CLEANING SUMMARY")
        print("="*80)
        print(f"Original Shape: {report.original_shape[0]:,} rows × {report.original_shape[1]} columns")
        print(f"Cleaned Shape:  {report.cleaned_shape[0]:,} rows × {report.cleaned_shape[1]} columns")
        print(f"Rows Removed:   {report.quality_metrics['rows_removed']:,}")
        print(f"Rows Retained:  {report.quality_metrics['rows_retained_percent']:.2f}%")
        print(f"Processing Time: {report.processing_time_seconds:.2f} seconds")
        
        if report.removal_breakdown:
            print("\nRemoval Breakdown:")
            for reason, count in report.removal_breakdown.items():
                print(f"  • {reason}: {count:,}")
        
        print("\nQuality Metrics:")
        print(f"  • Initial Missing: {report.quality_metrics['initial_missing_percent']:.2f}%")
        print(f"  • Final Missing:   {report.quality_metrics['final_missing_percent']:.2f}%")
        print(f"  • Duplicates Removed: {report.quality_metrics['duplicate_reduction']}")
        
        if report.warnings:
            print(f"\n⚠️  Warnings ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"    {warning}")
        
        if report.errors:
            print(f"\n❌ Errors ({len(report.errors)}):")
            for error in report.errors:
                print(f"    {error}")
        
        print("\n" + "="*80)
        print("✓ Data cleaning completed successfully!")
        print(f"✓ Cleaned data saved to: {output_dir / config.output_filename}")
        print(f"✓ Report saved to: {output_dir / config.report_output_path}")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        print("Please ensure the data file exists at the specified path.")
    except ValueError as e:
        print(f"\n❌ Error: Validation failed - {e}")
    except Exception as e:
        print(f"\n❌ Error: An unexpected error occurred - {e}")
        logger.exception("Unexpected error during execution")

