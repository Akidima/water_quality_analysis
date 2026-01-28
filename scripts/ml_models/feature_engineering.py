"""
Feature Engineering Module
Purpose: Data validation and preprocessing with Pydantic validation and support for Pandas/Dask DataFrames.
"""

import pandas as pd
from typing import List, Union, TYPE_CHECKING, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
from .model_utils import get_logger

if TYPE_CHECKING:
    import dask.dataframe as dd
    DaskDataFrame = dd.DataFrame
    DASK_AVAILABLE: bool
else:
    try:
        import dask.dataframe as dd
        DASK_AVAILABLE = True
        DaskDataFrame = dd.DataFrame
    except ImportError:
        DASK_AVAILABLE = False
        dd = None
        DaskDataFrame = Any

logger = get_logger(__name__)


class FeatureEngineeringConfig(BaseModel):
    """
    Configuration for feature engineering operations with Pydantic validation.
    
    Provides type-safe configuration with automatic validation for:
    - Required columns
    - Missing value handling strategy
    - DataFrame type preference
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    required_columns: List[str] = Field(
        ...,
        min_length=1,
        description="List of required column names for validation"
    )
    drop_missing: bool = Field(
        default=True,
        description="Whether to drop rows with missing values (True) or raise error (False)"
    )
    use_dask: bool = Field(
        default=False,
        description="Whether to prefer Dask DataFrame for large datasets"
    )
    min_rows_after_cleaning: int = Field(
        default=1,
        ge=0,
        description="Minimum number of rows required after cleaning (must be >= 0)"
    )
    
    @field_validator('required_columns')
    @classmethod
    def validate_required_columns(cls, v: List[str]) -> List[str]:
        """Validate required columns list is not empty and contains no duplicates."""
        if not v:
            raise ValueError("required_columns list cannot be empty")
        if len(v) != len(set(v)):
            raise ValueError("required_columns list contains duplicates")
        # Strip whitespace and filter empty strings
        cleaned = [col.strip() for col in v if col.strip()]
        if not cleaned:
            raise ValueError("required_columns list contains only empty strings")
        return cleaned


class ValidationResult(BaseModel):
    """
    Result of data validation with Pydantic validation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    is_valid: bool = Field(..., description="Whether validation passed")
    missing_columns: List[str] = Field(
        default_factory=list,
        description="List of missing required columns"
    )
    columns_with_missing_values: List[str] = Field(
        default_factory=list,
        description="List of columns containing missing values"
    )
    rows_dropped: int = Field(
        default=0,
        ge=0,
        description="Number of rows dropped during cleaning"
    )
    final_row_count: int = Field(
        default=0,
        ge=0,
        description="Final number of rows after cleaning"
    )
    initial_row_count: int = Field(
        default=0,
        ge=0,
        description="Initial number of rows before cleaning"
    )
    
    @field_validator('missing_columns', 'columns_with_missing_values')
    @classmethod
    def validate_string_lists(cls, v: List[str]) -> List[str]:
        """Ensure no empty strings in lists."""
        return [item for item in v if item.strip()]


def _is_dask_dataframe(df) -> bool:
    """Check if DataFrame is a Dask DataFrame."""
    if not DASK_AVAILABLE or dd is None:
        return False
    return isinstance(df, dd.DataFrame)


def _get_dataframe_length(df: Union[pd.DataFrame, DaskDataFrame]) -> int:
    """
    Safely get the length of a DataFrame (pandas or Dask).
    
    For Dask DataFrames, this computes the length efficiently.
    """
    if _is_dask_dataframe(df):
        return len(df)
    return len(df)


def _compute_if_dask(df: Union[pd.DataFrame, DaskDataFrame]) -> pd.DataFrame:
    """
    Convert Dask DataFrame to Pandas if needed, otherwise return as-is.
    
    Args:
        df: Either a Pandas or Dask DataFrame
        
    Returns:
        Pandas DataFrame
    """
    if _is_dask_dataframe(df):
        logger.info("Converting Dask DataFrame to Pandas DataFrame for processing")
        return df.compute()
    return df


class FeatureEngineer:
    """
    Handles data validation and preprocessing logic with Pydantic validation.
    
    Supports both Pandas and Dask DataFrames for flexible data processing.
    """

    @staticmethod
    def validate_data(
        df: Union[pd.DataFrame, DaskDataFrame],
        required_cols: List[str],
        drop_missing: bool = True,
        min_rows_after_cleaning: int = 1
    ) -> pd.DataFrame:
        """
        Ensures data has required columns and handles missing values.
        
        Args:
            df: Input DataFrame (Pandas or Dask)
            required_cols: List of required column names
            drop_missing: If True, drop rows with missing values; if False, raise error
            min_rows_after_cleaning: Minimum rows required after cleaning
            
        Returns:
            Cleaned Pandas DataFrame
            
        Raises:
            ValueError: If validation fails or data is invalid
            ValidationError: If Pydantic validation fails
        """
        # Validate configuration using Pydantic
        try:
            config = FeatureEngineeringConfig(
                required_columns=required_cols,
                drop_missing=drop_missing,
                min_rows_after_cleaning=min_rows_after_cleaning
            )
        except ValidationError as e:
            logger.error(f"Invalid configuration: {e}")
            raise
        
        # Convert Dask to Pandas if needed for validation checks
        is_dask = _is_dask_dataframe(df)
        initial_count = _get_dataframe_length(df)
        
        # 1. Check if columns exist
        missing = [col for col in config.required_columns if col not in df.columns]
        if missing:
            error_msg = f"Data is missing required columns: {missing}"
            logger.error(error_msg)
            result = ValidationResult(
                is_valid=False,
                missing_columns=missing,
                initial_row_count=initial_count
            )
            raise ValueError(error_msg)
        
        # 2. Check for missing values in required columns
        if is_dask:
            # For Dask, compute missing values check
            missing_values_series = df[config.required_columns].isnull().any()
            # Check if it's a Dask Series and compute if needed
            if DASK_AVAILABLE and isinstance(missing_values_series, dd.Series):
                missing_values_series = missing_values_series.compute()
            columns_with_missing = missing_values_series.index[missing_values_series].tolist()
        else:
            missing_values_series = df[config.required_columns].isnull().any()
            columns_with_missing = missing_values_series.index[missing_values_series].tolist()
        
        if columns_with_missing:
            if not config.drop_missing:
                error_msg = f"Data contains missing values in required columns: {columns_with_missing}"
                logger.error(error_msg)
                result = ValidationResult(
                    is_valid=False,
                    columns_with_missing_values=columns_with_missing,
                    initial_row_count=initial_count
                )
                raise ValueError(error_msg)
            else:
                logger.warning(f"Found missing values in columns: {columns_with_missing}. Will drop rows.")
        
        # 3. Subset data to relevant columns and handle missing values
        if is_dask:
            clean_df = df[config.required_columns].copy()
            # For Dask, dropna() returns a Dask DataFrame
            clean_df = clean_df.dropna()
            # Convert to Pandas for return
            clean_df = clean_df.compute()
        else:
            clean_df = df[config.required_columns].copy()
            clean_df = clean_df.dropna()
        
        final_count = len(clean_df)
        dropped_count = initial_count - final_count

        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows containing missing values.")
        
        if final_count < config.min_rows_after_cleaning:
            error_msg = (
                f"DataFrame has {final_count} rows after cleaning, "
                f"which is less than the minimum required ({config.min_rows_after_cleaning})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create validation result for logging
        result = ValidationResult(
            is_valid=True,
            columns_with_missing_values=columns_with_missing,
            rows_dropped=dropped_count,
            final_row_count=final_count,
            initial_row_count=initial_count
        )
        logger.info(
            f"Validation successful: {result.final_row_count} rows remaining "
            f"({result.rows_dropped} dropped)"
        )
        
        return clean_df
    
    @staticmethod
    def validate_data_with_config(
        df: Union[pd.DataFrame, DaskDataFrame],
        config: FeatureEngineeringConfig
    ) -> pd.DataFrame:
        """
        Validate data using a FeatureEngineeringConfig object.
        
        Args:
            df: Input DataFrame (Pandas or Dask)
            config: FeatureEngineeringConfig object with validation settings
            
        Returns:
            Cleaned Pandas DataFrame
        """
        return FeatureEngineer.validate_data(
            df=df,
            required_cols=config.required_columns,
            drop_missing=config.drop_missing,
            min_rows_after_cleaning=config.min_rows_after_cleaning
        )

