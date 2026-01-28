"""
Data Loading Module for Water Quality Analysis
Handles data ingestion with validation, error handling, and production-ready features.
"""

import dask.dataframe as dd
import numpy as np
import logging
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Configure module logger
logger = logging.getLogger(__name__)

# Import Pydantic enhancements
try:
    from pydantic_enhancements import (
        ColumnNameValidator, NumericStatistics, CategoricalStatistics,
        DatasetStatistics, ErrorHandler, ColumnValidationError,
        RangeValidationError, DataQualityError
    )
except ImportError:
    logger.warning("Pydantic enhancements module not available. Using basic features only.")
    ColumnNameValidator = None
    NumericStatistics = None
    CategoricalStatistics = None
    DatasetStatistics = None
    ErrorHandler = None
    ColumnValidationError = Exception
    RangeValidationError = Exception
    DataQualityError = Exception

class DataConfig(BaseModel):
    """Configuration for data loading with Pydantic validation.

        Think of this like a **recipe card** that tells us exactly how to
        prepare our ingredients (data) before cooking (analysis).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    filepath: str = Field(
        default='national_water_plan.csv',
        description="Path to the CSV file to load"
    )
    na_values: List[str] = Field(
        default_factory=lambda: ['', 'TBC', 'Pending Investigation', 'N/A', 
                                 'Unknown', 'Not Available', 'null', 'NULL'],
        description="List of strings to recognize as NA/null values"
    )
    chunk_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of rows to read at a time for chunked loading"
    )
    max_memory_mb: int = Field(
        default=500,
        gt=0,
        le=10000,
        description="Maximum memory usage in MB before switching to chunked loading"
    )
    required_columns: Optional[List[str]] = Field(
        default=None,
        description="List of columns that must be present in the dataset"
    )
    usecols: Optional[List[str]] = Field(
        default=None,
        description="List of columns to load from the CSV file"
    )
    dtype_optimization: bool = Field(
        default=True,
        description="Whether to optimize data types for memory efficiency"
    )

    @field_validator('filepath')
    @classmethod
    def validate_filepath(cls, v: str) -> str:
        """Validate that filepath is not empty and has valid extension."""
        if not v or not v.strip():
            raise ValueError("Filepath cannot be empty")
        
        # Check for valid CSV extension
        if not v.lower().endswith('.csv'):
            raise ValueError("Filepath must point to a CSV file (.csv extension)")
        
        return v.strip()
    
    @field_validator('na_values')
    @classmethod
    def validate_na_values(cls, v: List[str]) -> List[str]:
        """Ensure na_values list is not empty."""
        if not v:
            raise ValueError("na_values list cannot be empty")
        return v
    
    @model_validator(mode='after')
    def validate_column_consistency(self):
        """Validate that required_columns is a subset of usecols if both are specified."""
        if self.required_columns and self.usecols:
            required_set = set(self.required_columns)
            usecols_set = set(self.usecols)
            if not required_set.issubset(usecols_set):
                missing = required_set - usecols_set
                raise ValueError(
                    f"Required columns {missing} must be included in usecols"
                )
        return self

class ValidationMetadata(BaseModel):
    """Pydantic model for validation metadata."""
    rows: int = Field(ge=0, description="Number of rows in the dataset")
    columns: int = Field(ge=0, description="Number of columns in the dataset")
    memory_usage: float = Field(ge=0.0, description="Memory usage in MB")


class ValidationStats(BaseModel):
    """Pydantic model for validation statistics."""
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    memory_usage_mb: float = Field(ge=0.0)
    missing_values_percent: float = Field(ge=0.0, le=100.0)
    duplicate_rows: int = Field(ge=0)
    missing_values: Dict[str, int] = Field(default_factory=dict)
    data_types: Dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Pydantic model for comprehensive validation report."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: ValidationMetadata
    stats: Optional[ValidationStats] = None


class ExplorationMetadata(BaseModel):
    """Pydantic model for exploration metadata."""
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    memory_usage: float = Field(ge=0.0)
    missing_values_percent: float = Field(ge=0.0, le=100.0)
    duplicate_rows: int = Field(ge=0)


class ExplorationReport(BaseModel):
    """Pydantic model for comprehensive exploration report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    start_time: str
    end_time: Optional[str] = None
    duration: Optional[float] = Field(default=None, ge=0.0)
    config: Dict[str, Any]
    validation: Optional[ValidationReport] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: ExplorationMetadata


def _is_dask_dataframe(df) -> bool:
    """Check if DataFrame is a Dask DataFrame."""
    return isinstance(df, dd.DataFrame)


def _get_dataframe_length(df) -> int:
    """
    Safely get the length of a DataFrame (pandas or Dask).
    
    For Dask DataFrames, this computes the length efficiently.
    For pandas DataFrames, returns len() directly.
    """
    if _is_dask_dataframe(df):
        # For Dask, use shape[0].compute() which is more efficient than len()
        return int(df.shape[0].compute())
    else:
        return len(df)


def _safe_compute(obj):
    """
    Safely compute a Dask object if needed, otherwise return as-is.
    
    Args:
        obj: Either a Dask object (with .compute() method) or a regular object
        
    Returns:
        Computed value or original object
    """
    if hasattr(obj, 'compute'):
        return obj.compute()
    return obj


class DataValidator:
    """
    Validates data quality and integrity.

    Think of this like a **quality inspector at a toy factory** whoc checks 
    each toy to make sure it's safe and works properly before it goes to the store.
    """
    
    def __init__(self):
        """Initialize validator with error handler if available."""
        self.error_handler = ErrorHandler() if ErrorHandler else None

    @staticmethod
    def validate_file_exists(filepath: str) -> Path:
        """
        Check if the file exists and is accessible.

        Like checking if your toy box is actually in your room before trying to open it.
        """
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {filepath}\n"
                f"Current directory: {Path.cwd()}"
                )

        if not file_path.is_file():
            raise ValueError(f"Path exists but is not a file: {filepath}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"File is empty: {filepath}")
        
        logger.info(f"File validated: {filepath} ({file_path.stat().st_size / 1e6:.2f} MB)")
        return file_path

    def validate_dataframe(
        self,
        df: dd.DataFrame,
        min_rows: int = 1,
        required_columns: Optional[List[str]] = None,
    ) -> ValidationReport:
        """
        Validate a DataFrame for quality and integrity.

        Like a teacher checking your homework to make sure you:
        1. Did write something (not empty)
        2. Answered all the required columnns (no missing values)
        3. Wrote neatly enough to read (data is valid)
        """ 
        errors = []
        warnings = []
        is_valid = True
        
        # Initialize error handler if available
        if self.error_handler:
            self.error_handler.clear()
        
        # Validate column names using ColumnNameValidator if available
        if ColumnNameValidator:
            try:
                column_validation = ColumnNameValidator.validate_columns(df.columns.tolist())
                if column_validation.get('warnings'):
                    for col in column_validation['warnings']:
                        if self.error_handler:
                            self.error_handler.add_warning(f"Column '{col}' does not match standard patterns")
                        warnings.append(f"Column '{col}' does not match standard patterns")
                if column_validation.get('suggestions'):
                    for suggestion_info in column_validation['suggestions']:
                        suggestion_msg = f"Column '{suggestion_info['column']}': Consider {suggestion_info['suggestions']}"
                        if self.error_handler:
                            self.error_handler.add_warning(suggestion_msg)
                        warnings.append(suggestion_msg)
            except Exception as e:
                logger.warning(f"Column name validation failed: {e}")
        
        # Compute length for Dask DataFrame safely
        df_len = _get_dataframe_length(df)
        
        # Compute memory usage safely
        mem_usage_sum = df.memory_usage(deep=True).sum()
        mem_usage_mb = _safe_compute(mem_usage_sum) / 1e6
        
        metadata = ValidationMetadata(
            rows=df_len,
            columns=len(df.columns),
            memory_usage=mem_usage_mb
        )

        # Check if DataFrame is empty
        if df_len == 0 or df_len < min_rows:
            is_valid = False
            error_msg = f"DataFrame is empty or has less than {min_rows} rows"
            if self.error_handler and DataQualityError:
                self.error_handler.add_error(DataQualityError(
                    issue=error_msg,
                    affected_rows=df_len,
                    total_rows=min_rows,
                    threshold=0.0
                ))
            errors.append(error_msg)
            return ValidationReport(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )

        # Check for missing values in required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                is_valid = False
                for col in missing_cols:
                    error_msg = f"Missing required column: {col}"
                    if self.error_handler and ColumnValidationError:
                        self.error_handler.add_error(ColumnValidationError(
                            column_name=col,
                            issue="Required column is missing",
                            suggestion="Ensure the column exists in the dataset"
                        ))
                    errors.append(error_msg)
        
        # Check for completely empty columns
        null_check = df.isnull().all()
        null_check = _safe_compute(null_check)
        empty_cols = df.columns[null_check].tolist()
        if empty_cols:
            warning_msg = f"Columns with all missing values: {empty_cols}"
            if self.error_handler:
                for col in empty_cols:
                    self.error_handler.add_warning(f"Column '{col}' has all missing values")
            warnings.append(warning_msg)
        
        # Calculate statistics
        null_sum = df.isnull().sum()
        null_sum = _safe_compute(null_sum)
        missing_values_dict = {k: int(v) for k, v in null_sum.to_dict().items()}
        data_types_dict = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        
        total_nulls = null_sum.sum()
        missing_values_percent = (total_nulls / (df_len * len(df.columns))) * 100 if df_len > 0 else 0.0
        
        # Calculate duplicates (use map_partitions for Dask compatibility)
        try:
            if _is_dask_dataframe(df):
                # For Dask DataFrames, use map_partitions
                dup_sum = df.map_partitions(lambda x: x.duplicated().sum(), meta=('x', 'i8')).sum()
                dup_sum = _safe_compute(dup_sum)
            else:
                dup_sum = df.duplicated().sum()
        except Exception as e:
            # If duplicated check fails, set to 0 with warning
            logger.warning(f"Duplicate check failed: {e}")
            dup_sum = 0
            warnings.append("Duplicate check skipped (not supported for this operation)")
        
        mem_usage = df.memory_usage(deep=True).sum()
        mem_usage = _safe_compute(mem_usage)
        
        stats = ValidationStats(
            total_rows=df_len,
            total_columns=len(df.columns),
            memory_usage_mb=mem_usage / 1e6,
            missing_values_percent=missing_values_percent,
            duplicate_rows=int(dup_sum),
            missing_values=missing_values_dict,
            data_types=data_types_dict
        )

        # Warn about high missing data percentage using DataQualityError if available
        if stats.missing_values_percent > 0.5:
            warning_msg = f"High missing data percentage: {stats.missing_values_percent:.2f}%"
            if self.error_handler and DataQualityError:
                total_nulls = sum(missing_values_dict.values())
                self.error_handler.add_error(DataQualityError(
                    issue="High missing data percentage detected",
                    affected_rows=int(total_nulls),
                    total_rows=df_len,
                    threshold=0.5
                ))
            warnings.append(warning_msg)

        # Check for duplicate rows
        if stats.duplicate_rows > 0:
            warning_msg = f"Duplicate rows found: {stats.duplicate_rows}"
            if self.error_handler and DataQualityError:
                self.error_handler.add_error(DataQualityError(
                    issue="Duplicate rows detected",
                    affected_rows=int(dup_sum),
                    total_rows=df_len,
                    threshold=0.0
                ))
            warnings.append(warning_msg)
        
        # Include error handler report in warnings if available
        if self.error_handler and self.error_handler.has_warnings():
            error_report = self.error_handler.get_error_report()
            if error_report.get('warnings'):
                warnings.extend(error_report['warnings'])
        
        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
            stats=stats
        )

class DataLoader:
    """
    Production-ready data loader with validation and optimization features.

    Think of this like a **smart delivery truck** that:
    1. Checks if the delivery (file) is on time (exists and is accessible)
    2. Validates the contents (data quality and integrity)
    3. Optimizes the delivery (memory usage and performance)
    4. Reports any issues (errors and warnings)
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config if config else DataConfig()
        self.validator = DataValidator()

    def optimize_dtypes(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Optimize DataFrame memory usage by converting columns to more memory-efficient data types.
        
        Like organzing your closet by folding clothes tightly to fit more:
        - Big numbers become smaller numbers (int64 -> int32)
        - Long words become short codes (object -> category)
        - Empty space become empty (float64 -> float32)
        """
        initial_mem = df.memory_usage(deep=True).sum()
        initial_memory_usage = _safe_compute(initial_mem) / 1e6

        for col in df.columns:
            col_type = df[col].dtype

            # Optimize integer columns
            if col_type == 'int64':
                c_min = df[col].min()
                c_max = df[col].max()
                c_min = _safe_compute(c_min)
                c_max = _safe_compute(c_max)

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            # Optimize floats
            elif col_type == 'float64':
                df[col] = df[col].astype(np.float32)
            
            # Convert to category for low-cardinality object columns
            elif col_type == 'object':
                num_unique = df[col].nunique()
                num_unique_values = _safe_compute(num_unique)
                # For Series, use len() directly or compute if Dask Series
                col_series = df[col]
                if _is_dask_dataframe(df):
                    num_total = int(col_series.shape[0].compute())
                else:
                    num_total = len(col_series)
                
                if num_total > 0 and num_unique_values / num_total < 0.5: # Less than 50% unique values
                    df[col] = df[col].astype('category')

        final_mem = df.memory_usage(deep=True).sum()
        final_memory_usage = _safe_compute(final_mem) / 1e6
        
        reduction = (initial_memory_usage - final_memory_usage) / initial_memory_usage * 100 if initial_memory_usage > 0 else 0
        logger.info(
            f"Memory optimized: {initial_memory_usage:.2f} MB -> {final_memory_usage:.2f} MB "
            f"({reduction:.1f}% reduction)"
        )
        return df

    def load_data_chunked(self) -> dd.DataFrame:
        """
        Load data in chunks to optimize memory usage.

        Like a eating a big sandwich; instead of trying to fit the whole thing in your mouth at once,
        you take small bites (which won't work). This way you can enjoy the sandwich without getting too full.
        
        Note: Dask automatically handles chunking, so we just specify blocksize.
        """
        logger.info(f"Loading data with Dask (blocksize based on chunk_size)")

        try:
            # Dask handles chunking automatically with blocksize parameter
            # Convert chunk_size (rows) to approximate blocksize (bytes)
            # Assuming average row size of ~1KB
            blocksize = f"{self.config.chunk_size}KB" if self.config.chunk_size else '64MB'
            
            # Handle mixed dtypes by specifying object for problematic columns
            df = dd.read_csv(
                self.config.filepath,
                na_values=self.config.na_values,
                usecols=self.config.usecols,
                blocksize=blocksize,
                assume_missing=True,
                sample=256000,  # Sample more rows for better dtype inference
                dtype={'Non-bathing Priority Site Flag': 'object'}  # Handle mixed types
            )

            if self.config.dtype_optimization:
                df = self.optimize_dtypes(df)
            
            logger.info(f"Data loaded into Dask DataFrame with {df.npartitions} partitions")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
        finally:
            logger.info("Finished loading data")

    def load_and_explore_data(self) -> Tuple[dd.DataFrame, ExplorationReport]:
        """
        Load and explore data with comprehensive error handling and validation.

        Returns:
            Tuple of [DataFrame, ExplorationReport]
        
        Think of this like a **detective investigating a case**:
        1. First, they check if the crime scene (file) is accessible (exists and is accessible)
        2. Then, they collect evidence (data) and analyze it for clues (errors and warnings)
        3. Finally, they compile their findings into a report (exploration_report)
        """
        
        start_time = datetime.now()
        errors = []
        warnings = []

        try:
            # Step 1: Check if file exists
            logger.info(f"Starting data load from {self.config.filepath}")
            file_path = self.validator.validate_file_exists(self.config.filepath)
            
            # Step 2: Determine loading strategy
            file_size_mb = file_path.stat().st_size / 1e6
            
            if self.config.chunk_size or file_size_mb > self.config.max_memory_mb:
                if not self.config.chunk_size:
                    self.config.chunk_size = 10000
                    warnings.append(
                        f"Large file detected. ({file_size_mb:.2f} MB). "
                        f"Using chunk loading with chunk_size={self.config.chunk_size}"
                    )
                df = self.load_data_chunked()
            else:
                # Load entire file at once
                logger.info(f"Loading entire file at once ({file_size_mb:.2f} MB)")
                # Handle mixed dtypes by specifying object for problematic columns
                df = dd.read_csv(
                    self.config.filepath,
                    na_values=self.config.na_values,
                    usecols=self.config.usecols,
                    assume_missing=True,
                    sample=256000,  # Sample more rows for better dtype inference
                    dtype={'Non-bathing Priority Site Flag': 'object'}  # Handle mixed types
                )

                if self.config.dtype_optimization:
                    df = self.optimize_dtypes(df)
            
            # Step 3: Validate loaded data
            logger.info("Validating data")
            validation_report = self.validator.validate_dataframe(
                df,
                required_columns=self.config.required_columns
                )

            if not validation_report.is_valid:
                raise ValueError(
                    f"Data validation failed: {validation_report.errors}"
                )
            
            # Step 4: Generate exploration statistics
            logger.info("Generating exploration statistics")
            statistics = self._generate_statistics(df)
            
            # Step 5: Build exploration report
            logger.info("Data load completed")
            load_time = (datetime.now() - start_time).total_seconds()
            
            # Combine warnings from validation and loading
            all_warnings = warnings + validation_report.warnings
            all_errors = errors + validation_report.errors
            
            # Create metadata - compute values for Dask
            df_len = _get_dataframe_length(df)
            
            null_sum = df.isnull().sum().sum()
            null_sum = _safe_compute(null_sum)
            missing_values_percent = (null_sum / (df_len * len(df.columns))) * 100 if df_len > 0 else 0.0
            
            mem_usage = df.memory_usage(deep=True).sum()
            mem_usage = _safe_compute(mem_usage)
            
            # Calculate duplicates (Dask-compatible)
            try:
                if _is_dask_dataframe(df):
                    dup_count = df.map_partitions(lambda x: x.duplicated().sum(), meta=('x', 'i8')).sum()
                    dup_count = _safe_compute(dup_count)
                else:
                    dup_count = df.duplicated().sum()
            except Exception as e:
                logger.warning(f"Duplicate check failed: {e}")
                dup_count = 0
            
            metadata = ExplorationMetadata(
                rows=df_len,
                columns=len(df.columns),
                memory_usage=mem_usage / 1e6,
                missing_values_percent=missing_values_percent,
                duplicate_rows=int(dup_count)
            )
            
            # Create exploration report
            exploration_report = ExplorationReport(
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=load_time,
                config=self.config.model_dump(),
                validation=validation_report,
                statistics=statistics,
                errors=all_errors,
                warnings=all_warnings,
                metadata=metadata
            )
            
            logger.info(f"Data loaded successfully: {metadata.model_dump()}")
            
            # Log warnings
            for warning in all_warnings:
                logger.warning(warning)
            
            # Log errors
            for error in all_errors:
                logger.error(error)
            
            return df, exploration_report
        
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise
        except ValueError as e:
            if "empty" in str(e).lower():
                logger.error("CSV file is empty or corrupted")
                raise ValueError("CSV file contains no data")
            elif "parse" in str(e).lower() or "CSV" in str(e):
                logger.error(f"CSV parsing error: {e}")
                raise ValueError(f"Invalid CSV format: {str(e)}")
            else:
                raise
        except MemoryError:
            logger.error("Memory error: Not enough memory to load data")
            raise MemoryError(
                "Not enough memory to load data. Try reducing chunk size or using a smaller file. "
                "Try increasing available memory or using a smaller chunk size"
            )
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}", exc_info=True)
            raise
    
    def _generate_statistics(self, df: dd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the dataset.
        
        Returns:
            Dict[str, Any]: Dictionary containing various statistics about the data
        
        Like a doctor doing a health check-up, this function performs a thorough analysis of the dataset,
        - Your data is like a patient, and this function is like a doctor
        - It checks for missing values, duplicates, and other potential issues
        - It also calculates summary statistics, such as mean, median, and standard deviation
        """
        # Compute values for Dask DataFrames
        df_len = _get_dataframe_length(df)
        
        # Compute missing values
        missing_vals = df.isnull().sum()
        missing_vals = _safe_compute(missing_vals)
        
        missing_dict = missing_vals.to_dict()
        missing_pct_dict = {k: (v / df_len * 100) if df_len > 0 else 0 for k, v in missing_dict.items()}
        
        # Compute duplicates (Dask-compatible)
        try:
            if _is_dask_dataframe(df):
                dup_count = df.map_partitions(lambda x: x.duplicated().sum(), meta=('x', 'i8')).sum()
                dup_count = _safe_compute(dup_count)
            else:
                dup_count = df.duplicated().sum()
        except Exception as e:
            logger.warning(f"Duplicate check failed in statistics: {e}")
            dup_count = 0
        
        # Compute memory usage
        mem_usage = df.memory_usage(deep=True).sum()
        mem_usage = _safe_compute(mem_usage)
        mem_usage_mb = mem_usage / 1e6
        
        # Calculate overall missing percent
        total_nulls = sum(missing_dict.values())
        overall_missing_percent = (total_nulls / (df_len * len(df.columns))) * 100 if df_len > 0 else 0.0
        
        # Use DatasetStatistics if available
        numeric_stats_dict = {}
        categorical_stats_dict = {}
        
        if DatasetStatistics and NumericStatistics and CategoricalStatistics:
            # Generate numeric statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                try:
                    col_data = df[col]
                    col_data = _safe_compute(col_data)
                    
                    desc = col_data.describe()
                    missing_count = int(missing_dict.get(col, 0))
                    missing_pct = missing_pct_dict.get(col, 0.0)
                    
                    numeric_stats_dict[col] = NumericStatistics(
                        column_name=col,
                        count=int(df_len),
                        mean=float(desc.get('mean', 0)) if 'mean' in desc else None,
                        std=float(desc.get('std', 0)) if 'std' in desc else None,
                        min=float(desc.get('min', 0)) if 'min' in desc else None,
                        q25=float(desc.get('25%', 0)) if '25%' in desc else None,
                        median=float(desc.get('50%', 0)) if '50%' in desc else None,
                        q75=float(desc.get('75%', 0)) if '75%' in desc else None,
                        max=float(desc.get('max', 0)) if 'max' in desc else None,
                        missing_count=missing_count,
                        missing_percent=missing_pct,
                        outlier_count=0,  # Could be enhanced with outlier detection
                        outlier_percent=0.0
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate numeric statistics for {col}: {e}")
            
            # Generate categorical statistics
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            for col in categorical_columns:
                try:
                    col_data = df[col]
                    col_data = _safe_compute(col_data)
                    
                    nunique = col_data.nunique()
                    missing_count = int(missing_dict.get(col, 0))
                    missing_pct = missing_pct_dict.get(col, 0.0)
                    
                    mode_val = col_data.mode()
                    top_value = mode_val.iloc[0] if len(mode_val) > 0 else None
                    top_frequency = int(col_data.value_counts().iloc[0]) if len(col_data.value_counts()) > 0 else 0
                    
                    categorical_stats_dict[col] = CategoricalStatistics(
                        column_name=col,
                        count=int(df_len),
                        unique_count=int(nunique),
                        unique_percent=(nunique / df_len * 100) if df_len > 0 else 0.0,
                        top_value=str(top_value) if top_value is not None else None,
                        top_frequency=top_frequency,
                        missing_count=missing_count,
                        missing_percent=missing_pct
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate categorical statistics for {col}: {e}")
            
            # Create DatasetStatistics
            try:
                dataset_stats = DatasetStatistics(
                    total_rows=df_len,
                    total_columns=len(df.columns),
                    numeric_columns=numeric_stats_dict,
                    categorical_columns=categorical_stats_dict,
                    memory_usage_mb=mem_usage_mb,
                    overall_missing_percent=overall_missing_percent,
                    duplicate_rows=int(dup_count)
                )
                # Include dataset statistics summary in return
                stats = {
                    'shape': {'rows': df_len, 'columns': len(df.columns)},
                    'columns': df.columns.to_list(),
                    'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
                    'missing_values': missing_dict,
                    'missing_percentage': missing_pct_dict,
                    'duplicate_rows': int(dup_count),
                    'dataset_statistics': dataset_stats.model_dump(),
                    'quality_score': dataset_stats.overall_quality_score(),
                    'summary_report': dataset_stats.summary_report()
                }
            except Exception as e:
                logger.warning(f"Failed to create DatasetStatistics: {e}")
                # Fall back to basic stats
                stats = self._generate_basic_statistics(df, df_len, missing_dict, missing_pct_dict, dup_count, mem_usage_mb)
        else:
            # Fall back to basic statistics if enhancements not available
            stats = self._generate_basic_statistics(df, df_len, missing_dict, missing_pct_dict, dup_count, mem_usage_mb)
        
        return stats
    
    def _generate_basic_statistics(self, df: dd.DataFrame, df_len: int, missing_dict: Dict, 
                                  missing_pct_dict: Dict, dup_count: int, mem_usage_mb: float) -> Dict[str, Any]:
        """Generate basic statistics without pydantic enhancements."""
        # Compute describe statistics
        desc_stats = df.describe()
        desc_stats = _safe_compute(desc_stats)
        
        stats = {
            'shape': {'rows': df_len, 'columns': len(df.columns)},
            'columns': df.columns.to_list(),
            'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
            'missing_values': missing_dict,
            'missing_percentage': missing_pct_dict,
            'duplicate_rows': int(dup_count),
            'summary_statistics': desc_stats.to_dict()
        }

        # Missing values analysis
        stats['missing_values'] = {
            col: {
                'count': int(missing_dict[col]),
                'percentage': float(missing_pct_dict[col])
            }
            for col in missing_dict.keys() if missing_dict[col] > 0
        }

        # Numerical columns summary
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            numeric_desc = df.select_dtypes(include=[np.number]).describe()
            numeric_desc = _safe_compute(numeric_desc)
            stats['summary_statistics']['numeric'] = numeric_desc.to_dict()
        
        # Categorical columns summary
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_columns) > 0:
            stats['summary_statistics']['categorical'] = {}
            for col in categorical_columns:
                nunique = df[col].nunique()
                nunique = _safe_compute(nunique)
                
                mode_val = df[col].mode()
                mode_val = _safe_compute(mode_val)
                
                stats['summary_statistics']['categorical'][col] = {
                    'unique_values': int(nunique),
                    'unique_percentage': (nunique / df_len * 100) if df_len > 0 else 0,
                    'top_value': mode_val.iloc[0] if len(mode_val) > 0 else None,
                }
        return stats

    def save_exploration_report(
        self,
        report: ExplorationReport,
        output_path: str = 'data_exploration_report.json'
    ) -> None:
        """
        Save the exploration report to a JSON file.

        Like keeping a diary of what you discovered about your data, so you can read it later and remember all the important details.

        Args:
            report (ExplorationReport): The exploration report to save
            output_path (str, optional): The path to save the report to. Defaults to 'data_exploration_report.json'.
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(report.model_dump(), f, indent=4, default=str)
            logger.info(f"Exploration report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save exploration report: {str(e)}")
            raise

# Convenience function for backward compatibility
def load_and_explore_data(
    filepath: str = 'national_water_plan.csv',
    **kwargs
    ) -> Tuple[dd.DataFrame, ExplorationReport]:
    """
    Load and explore data from a CSV file with Pydantic validation.

    Args:
        filepath (str, optional): The path to the CSV file to load. Defaults to 'national_water_plan.csv'.
        **kwargs: Additional configuration parameters for DataConfig.

    Returns:
        Tuple[dd.DataFrame, ExplorationReport]: A tuple containing the loaded Dask DataFrame and the exploration report
    """
    config = DataConfig(filepath=filepath, **kwargs)
    loader = DataLoader(config)
    return loader.load_and_explore_data()


def main():
    """Main entry point for the data loader CLI."""
    parser = argparse.ArgumentParser(
        description='Load and explore water quality data with validation and statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load default data file
  python scripts/data-loader.py
  
  # Load specific file
  python scripts/data-loader.py --file data/my_data.csv
  
  # Load with custom chunk size
  python scripts/data-loader.py --file data/large_data.csv --chunk-size 10000
  
  # Load with custom output path
  python scripts/data-loader.py --file data/my_data.csv --output export/my_report.json
  
  # Load with verbose logging
  python scripts/data-loader.py --file data/my_data.csv --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help='Path to the CSV file to load (default: data/national_water_plan.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save the exploration report JSON (default: export/data_exploration_report.json)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Number of rows to read at a time for chunked loading (default: auto-detect)'
    )
    
    parser.add_argument(
        '--max-memory-mb',
        type=int,
        default=500,
        help='Maximum memory usage in MB before switching to chunked loading (default: 500)'
    )
    
    parser.add_argument(
        '--no-dtype-optimization',
        action='store_true',
        help='Disable data type optimization for memory efficiency'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--show-head',
        action='store_true',
        default=True,
        help='Show first 5 rows of the loaded data (default: True)'
    )
    
    parser.add_argument(
        '--no-show-head',
        dest='show_head',
        action='store_false',
        help='Do not show first 5 rows of the loaded data'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Determine input file path
    if args.file:
        data_path = Path(args.file)
        if not data_path.is_absolute():
            data_path = Path(__file__).parent.parent / args.file
    else:
        data_path = Path(__file__).parent.parent / 'data' / 'national_water_plan.csv'
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path(__file__).parent.parent / args.output
    else:
        output_path = Path(__file__).parent.parent / 'export' / 'data_exploration_report.json'
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {data_path}")
    
    # Create configuration
    config_kwargs = {
        'filepath': str(data_path),
        'dtype_optimization': not args.no_dtype_optimization,
        'max_memory_mb': args.max_memory_mb
    }
    
    if args.chunk_size:
        config_kwargs['chunk_size'] = args.chunk_size
    
    # Load and explore the data
    try:
        df, report = load_and_explore_data(**config_kwargs)
        
        # Print summary
        print("\n" + "="*80)
        print("DATA LOADING SUMMARY")
        print("="*80)
        print(f"Rows: {report.metadata.rows:,}")
        print(f"Columns: {report.metadata.columns}")
        print(f"Memory Usage: {report.metadata.memory_usage:.2f} MB")
        print(f"Missing Values: {report.metadata.missing_values_percent:.2f}%")
        print(f"Duplicate Rows: {report.metadata.duplicate_rows}")
        print(f"Duration: {report.duration:.2f} seconds")
        
        if report.warnings:
            print(f"\nWarnings ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  ⚠️  {warning}")
        
        if report.errors:
            print(f"\nErrors ({len(report.errors)}):")
            for error in report.errors:
                print(f"  ❌ {error}")
        
        if args.show_head:
            print("\n" + "="*80)
            print(f"First 5 rows:")
            print("="*80)
            # Dask's head() already returns computed pandas DataFrame
            print(df.head())
        
        # Save the report
        loader = DataLoader()
        loader.save_exploration_report(report, str(output_path))
        print(f"\n📄 Report saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())