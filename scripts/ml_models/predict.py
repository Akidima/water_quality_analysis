"""
Prediction Pipeline Module
Purpose: Prediction pipeline for spill prediction models with Pydantic validation 
         and support for Pandas/Dask DataFrames.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

from .model_utils import ModelConfig, get_logger
from .feature_engineering import FeatureEngineer
from .model_base import ModelBase, ModelMetrics, _ensure_pandas_dataframe, _is_dask_dataframe

# Import enhanced Pydantic validators
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pydantic_enhancements import DataFrameInputValidator

logger = get_logger(__name__)


class PredictionOutput(BaseModel):
    """
    Pydantic model for validating prediction output.
    
    Ensures predictions are in the correct format.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    predictions: List[float] = Field(
        ...,
        description="List of prediction values"
    )
    sample_count: int = Field(
        ...,
        ge=0,
        description="Number of samples predicted"
    )
    
    @field_validator('predictions')
    @classmethod
    def validate_predictions(cls, v: List[float]) -> List[float]:
        """Validate predictions are numeric."""
        if not isinstance(v, list):
            raise ValueError("Predictions must be a list")
        validated = []
        for i, pred in enumerate(v):
            if not isinstance(pred, (int, float, np.number)):
                raise ValueError(f"Prediction at index {i} must be numeric, got {type(pred)}")
            validated.append(float(pred))
        return validated
    
    @model_validator(mode='after')
    def validate_sample_count_matches_predictions(self) -> 'PredictionOutput':
        """Validate sample count matches predictions length."""
        if len(self.predictions) != self.sample_count:
            raise ValueError(
                f"Sample count ({self.sample_count}) does not match predictions length ({len(self.predictions)})"
            )
        return self


class SpillPredictionPipeline(ModelBase):
    """
    Prediction pipeline for spill prediction models.
    
    Supports:
    - Pandas DataFrames
    - Dask DataFrames (converted to Pandas for sklearn compatibility)
    - List of dictionaries (converted to DataFrame)
    - Pydantic validation for inputs and outputs
    """
    
    def __init__(self, config: Optional[ModelConfig] = None, model_name: Optional[str] = None):
        """
        Initialize the prediction pipeline.
        
        Args:
            config: ModelConfig instance with model configuration
            model_name: Optional name for the model (for logging)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            IOError: If model loading fails
        """
        super().__init__(model_name=model_name or "SpillPredictionPipeline")
        self.config = config or ModelConfig()
        self.pipeline = None
        self.metrics = ModelMetrics()
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Internal method to load the model artifact.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            IOError: If model loading fails
        """
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
        
        try:
            self.pipeline = joblib.load(self.config.model_path)
            self._set_trained(True)
            logger.info(f"Model loaded successfully from {self.config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.config.model_path}: {e}")
            raise IOError(f"Failed to load model: {e}") from e
    
    def predict(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], 'dd.DataFrame'],
        **kwargs
    ) -> List[float]:
        """
        Make predictions on new data with Pydantic validation.
        
        Args:
            data: Input data as pandas DataFrame, Dask DataFrame, or list of dictionaries
            **kwargs: Additional prediction parameters (unused, for compatibility)
            
        Returns:
            List of prediction values
            
        Raises:
            RuntimeError: If model is not trained or pipeline is None
            ValueError: If data validation fails
            ValidationError: If Pydantic validation fails
        """
        # Check if model is trained
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model is not trained or pipeline is not loaded. Cannot make predictions.")
        
        # Validate input using enhanced Pydantic validator
        try:
            input_validator = DataFrameInputValidator(
                data=data,
                required_columns=None,  # Will be validated by FeatureEngineer
                min_rows=1
            )
            validated_data = input_validator.data
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            raise
        
        # Convert list of dicts to DataFrame if necessary
        if isinstance(validated_data, list):
            try:
                validated_data = pd.DataFrame(validated_data)
                logger.debug("Converted list of dictionaries to pandas DataFrame")
            except Exception as e:
                logger.error(f"Failed to convert list to DataFrame: {e}")
                raise ValueError(f"Failed to convert input data to DataFrame: {e}") from e
        
        # Convert Dask DataFrame to Pandas if needed
        if _is_dask_dataframe(validated_data):
            validated_data = _ensure_pandas_dataframe(validated_data)
        
        # Validate features using FeatureEngineer
        try:
            clean_data = FeatureEngineer.validate_data(
                df=validated_data,
                required_cols=self.config.features,
                drop_missing=True,
                min_rows_after_cleaning=1
            )
        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            raise ValueError(f"Feature validation failed: {e}") from e
        
        logger.info(f"Making predictions for {len(clean_data)} samples...")
        
        # Make predictions
        try:
            predictions = self.pipeline.predict(clean_data)
            
            # Convert numpy array to list if needed
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
            elif not isinstance(predictions, list):
                predictions = list(predictions)
            
            # Validate output using Pydantic
            output_validator = PredictionOutput(
                predictions=predictions,
                sample_count=len(predictions)
            )
            
            logger.info(f"Successfully made predictions for {output_validator.sample_count} samples")
            return output_validator.predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    def train(
        self,
        data: Union[pd.DataFrame, 'dd.DataFrame'],
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the model (not implemented for prediction-only pipeline).
        
        Args:
            data: Training data (not used)
            **kwargs: Additional training parameters (not used)
            
        Raises:
            NotImplementedError: This pipeline is for prediction only
        """
        raise NotImplementedError(
            "SpillPredictionPipeline is for prediction only. "
            "Use SpillTrainingPipeline for training."
        )
    
    def save(self, filepath: Union[str, Path] = None) -> None:
        """
        Save the model artifact to disk.
        
        Args:
            filepath: Optional path where the model should be saved.
                     If None, uses config.model_path
            
        Raises:
            IOError: If saving fails
            ValueError: If filepath is invalid or model is not trained
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Cannot save: Model is not trained or pipeline is None")
        
        save_path = Path(filepath) if filepath else self.config.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(self.pipeline, save_path)
            logger.info(f"Model saved successfully to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {save_path}: {e}")
            raise IOError(f"Failed to save model: {e}") from e
    
    def load(self, filepath: Union[str, Path] = None) -> None:
        """
        Load the model artifact from disk.
        
        Args:
            filepath: Optional path to the saved model file.
                     If None, uses config.model_path
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            IOError: If loading fails
        """
        load_path = Path(filepath) if filepath else self.config.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Update config if custom path provided
        if filepath:
            self.config.model_filename = load_path.name
            self.config.artifacts_dir = load_path.parent
        
        self._load_model()


# Helper functions for quick script usage
def make_prediction(
    data: Union[pd.DataFrame, List[Dict[str, Any]], 'dd.DataFrame'],
    config: Optional[ModelConfig] = None
) -> List[float]:
    """
    Helper function for quick predictions.
    
    Args:
        data: Input data as pandas DataFrame, Dask DataFrame, or list of dictionaries
        config: Optional ModelConfig instance. If None, uses default config.
        
    Returns:
        List of prediction values
        
    Raises:
        RuntimeError: If model is not trained or prediction fails
        ValueError: If data validation fails
        FileNotFoundError: If model file doesn't exist
    """
    predictor = SpillPredictionPipeline(config=config)
    return predictor.predict(data)
