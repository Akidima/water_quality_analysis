"""
Model Base Module
Purpose: Abstract base class for ML models with Pydantic validation and support for Pandas/Dask DataFrames.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

from .model_utils import get_logger

# Import enhanced Pydantic validators
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pydantic_enhancements import ModelMetricsConfig

logger = get_logger(__name__)


def _is_dask_dataframe(df) -> bool:
    """Check if DataFrame is a Dask DataFrame."""
    if not DASK_AVAILABLE:
        return False
    return isinstance(df, dd.DataFrame)


def _ensure_pandas_dataframe(df: Union[pd.DataFrame, 'dd.DataFrame']) -> pd.DataFrame:
    """
    Convert Dask DataFrame to Pandas if needed, otherwise return as-is.
    
    Args:
        df: Either a Pandas or Dask DataFrame
        
    Returns:
        Pandas DataFrame
        
    Raises:
        TypeError: If input is not a DataFrame
    """
    if _is_dask_dataframe(df):
        logger.info("Converting Dask DataFrame to Pandas DataFrame for processing")
        return df.compute()
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise TypeError(f"Expected pandas or dask DataFrame, got {type(df)}")


class TrainingDataConfig(BaseModel):
    """
    Configuration for training data validation with Pydantic validation.
    
    Provides type-safe validation for:
    - Minimum required rows
    - Required columns
    - Data quality checks
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    min_rows: int = Field(
        default=1,
        ge=1,
        description="Minimum number of rows required for training (must be >= 1)"
    )
    required_columns: Optional[List[str]] = Field(
        default=None,
        description="Optional list of required column names for validation"
    )
    
    @field_validator('required_columns')
    @classmethod
    def validate_required_columns(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate required columns list contains no duplicates or empty strings."""
        if v is None:
            return v
        if not v:
            raise ValueError("required_columns list cannot be empty if provided")
        if len(v) != len(set(v)):
            raise ValueError("required_columns list contains duplicates")
        cleaned = [col.strip() for col in v if col.strip()]
        if not cleaned:
            raise ValueError("required_columns list contains only empty strings")
        return cleaned


class PredictionDataConfig(BaseModel):
    """
    Configuration for prediction data validation with Pydantic validation.
    
    Provides type-safe validation for:
    - Minimum required rows
    - Required columns
    - Data quality checks
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    min_rows: int = Field(
        default=1,
        ge=1,
        description="Minimum number of rows required for prediction (must be >= 1)"
    )
    required_columns: Optional[List[str]] = Field(
        default=None,
        description="Optional list of required column names for validation"
    )
    
    @field_validator('required_columns')
    @classmethod
    def validate_required_columns(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate required columns list contains no duplicates or empty strings."""
        if v is None:
            return v
        if not v:
            raise ValueError("required_columns list cannot be empty if provided")
        if len(v) != len(set(v)):
            raise ValueError("required_columns list contains duplicates")
        cleaned = [col.strip() for col in v if col.strip()]
        if not cleaned:
            raise ValueError("required_columns list contains only empty strings")
        return cleaned


class ModelMetrics(BaseModel):
    """
    Standardized model metrics with Pydantic validation.
    
    Provides type-safe storage for model performance metrics.
    Enhanced with ModelMetricsConfig for better validation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of metric names to values"
    )
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate metrics dictionary contains valid float values."""
        if not isinstance(v, dict):
            raise ValueError("metrics must be a dictionary")
        validated = {}
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"Metric key must be a string, got {type(key)}")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric value must be numeric, got {type(value)} for key '{key}'")
            validated[key] = float(value)
        return validated
    
    def to_metrics_config(self) -> ModelMetricsConfig:
        """
        Convert to ModelMetricsConfig for enhanced validation and analysis.
        
        Returns:
            ModelMetricsConfig instance with parsed metrics
        """
        # Map common metric names to ModelMetricsConfig fields
        metric_mapping = {
            'r2': 'r2_score',
            'r2_score': 'r2_score',
            'rmse': 'rmse',
            'mae': 'mae',
            'mse': 'mse',
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1_score',
            'f1_score': 'f1_score',
            'loss': 'loss'
        }
        
        config_dict = {}
        for key, value in self.metrics.items():
            key_lower = key.lower()
            if key_lower in metric_mapping:
                config_dict[metric_mapping[key_lower]] = value
        
        return ModelMetricsConfig(**config_dict)
    
    def is_good_performance(self, threshold: float = 0.7) -> bool:
        """
        Check if model performance meets threshold using enhanced validation.
        
        Args:
            threshold: Performance threshold (default: 0.7)
            
        Returns:
            True if performance meets threshold
        """
        try:
            metrics_config = self.to_metrics_config()
            return metrics_config.is_good_performance(threshold)
        except Exception as e:
            logger.warning(f"Could not evaluate performance using enhanced validator: {e}")
            # Fallback to simple check
            if 'r2' in self.metrics:
                return self.metrics['r2'] >= threshold
            elif 'accuracy' in self.metrics:
                return self.metrics['accuracy'] >= threshold
            return False


class ModelBase(ABC):
    """
    Abstract base class ensuring all models follow the same structure.
    
    Provides a consistent interface for model training, prediction, and persistence
    with Pydantic validation and support for both Pandas and Dask DataFrames.
    
    All subclasses must implement:
    - train(): Train the model on provided data
    - predict(): Make predictions on new data
    - save(): Save the model to disk
    - load(): Load the model from disk
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Optional name for the model (for logging and identification)
        """
        self.model_name = model_name or self.__class__.__name__
        self._is_trained = False
        logger.info(f"Initialized {self.model_name}")

    def _validate_training_data(
        self,
        data: Union[pd.DataFrame, 'dd.DataFrame'],
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1
    ) -> pd.DataFrame:
        """
        Validate training data using Pydantic validation.
        
        Args:
            data: Input DataFrame (Pandas or Dask)
            required_columns: Optional list of required column names
            min_rows: Minimum number of rows required
            
        Returns:
            Validated Pandas DataFrame
            
        Raises:
            ValueError: If validation fails
            ValidationError: If Pydantic validation fails
        """
        try:
            config = TrainingDataConfig(
                min_rows=min_rows,
                required_columns=required_columns
            )
        except ValidationError as e:
            logger.error(f"Invalid training data configuration: {e}")
            raise
        
        # Convert to Pandas for validation
        df = _ensure_pandas_dataframe(data)
        
        # Check minimum rows
        if len(df) < config.min_rows:
            raise ValueError(
                f"Training data has {len(df)} rows, "
                f"which is less than the minimum required ({config.min_rows})"
            )
        
        # Check required columns
        if config.required_columns:
            missing = [col for col in config.required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Training data is missing required columns: {missing}")
        
        logger.debug(f"Training data validation passed: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _validate_prediction_data(
        self,
        data: Union[pd.DataFrame, 'dd.DataFrame'],
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1
    ) -> pd.DataFrame:
        """
        Validate prediction data using Pydantic validation.
        
        Args:
            data: Input DataFrame (Pandas or Dask)
            required_columns: Optional list of required column names
            min_rows: Minimum number of rows required
            
        Returns:
            Validated Pandas DataFrame
            
        Raises:
            ValueError: If validation fails
            ValidationError: If Pydantic validation fails
        """
        try:
            config = PredictionDataConfig(
                min_rows=min_rows,
                required_columns=required_columns
            )
        except ValidationError as e:
            logger.error(f"Invalid prediction data configuration: {e}")
            raise
        
        # Convert to Pandas for validation
        df = _ensure_pandas_dataframe(data)
        
        # Check minimum rows
        if len(df) < config.min_rows:
            raise ValueError(
                f"Prediction data has {len(df)} rows, "
                f"which is less than the minimum required ({config.min_rows})"
            )
        
        # Check required columns
        if config.required_columns:
            missing = [col for col in config.required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Prediction data is missing required columns: {missing}")
        
        logger.debug(f"Prediction data validation passed: {len(df)} rows, {len(df.columns)} columns")
        return df

    @abstractmethod
    def train(
        self,
        data: Union[pd.DataFrame, 'dd.DataFrame'],
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the model on provided data.
        
        Args:
            data: Training data as Pandas or Dask DataFrame
            **kwargs: Additional training parameters specific to the model
            
        Returns:
            Dictionary of training metrics (e.g., {'accuracy': 0.95, 'loss': 0.05})
            
        Raises:
            ValueError: If data validation fails
            RuntimeError: If training fails
        """
        pass

    @abstractmethod
    def predict(
        self,
        data: Union[pd.DataFrame, 'dd.DataFrame'],
        **kwargs
    ) -> Union[pd.Series, pd.DataFrame, List[Any], Any]:
        """
        Make predictions on new data.
        
        Args:
            data: Prediction data as Pandas or Dask DataFrame
            **kwargs: Additional prediction parameters specific to the model
            
        Returns:
            Predictions as Series, DataFrame, List, or other appropriate type
            
        Raises:
            ValueError: If data validation fails
            RuntimeError: If model is not trained or prediction fails
        """
        pass

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the model artifact to disk.
        
        Args:
            filepath: Path where the model should be saved
            
        Raises:
            IOError: If saving fails
            ValueError: If filepath is invalid
        """
        pass

    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load the model artifact from disk.
        
        Args:
            filepath: Path to the saved model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            IOError: If loading fails
            ValueError: If filepath is invalid or model is corrupted
        """
        pass

    @property
    def is_trained(self) -> bool:
        """
        Check if the model has been trained.
        
        Returns:
            True if model is trained, False otherwise
        """
        return self._is_trained

    def _set_trained(self, value: bool = True) -> None:
        """
        Set the trained status of the model.
        
        Args:
            value: Whether the model is trained (default: True)
        """
        self._is_trained = value
        logger.debug(f"{self.model_name} trained status set to {value}")
