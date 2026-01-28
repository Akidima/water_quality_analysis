"""
Training Pipeline Module
Purpose: ML training pipeline with Pydantic validation and support for Pandas/Dask DataFrames.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Union
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pydantic import ValidationError

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

from .model_base import ModelBase, ModelMetrics
from .model_utils import ModelConfig, get_logger
from .feature_engineering import FeatureEngineer

# Import enhanced Pydantic validators
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pydantic_enhancements import ModelHyperparameterConfig

logger = get_logger(__name__)


class SpillTrainingPipeline(ModelBase):
    """
    Training pipeline for spill prediction models.
    
    Uses Pydantic validation for configuration and metrics, and supports
    both Pandas and Dask DataFrames (converted to Pandas for sklearn compatibility).
    """
    
    def __init__(self, config: ModelConfig = None, model_name: str = None):
        """
        Initialize the training pipeline.
        
        Args:
            config: ModelConfig instance with training configuration
            model_name: Optional name for the model (for logging)
        """
        super().__init__(model_name=model_name or "SpillTrainingPipeline")
        self.config = config or ModelConfig()
        self.pipeline = None
        self.metrics = ModelMetrics()
    
    def train(self, data: Union[pd.DataFrame, 'dd.DataFrame'], **kwargs) -> Dict[str, float]:
        """
        Train the model on provided data with Pydantic validation.
        
        Args:
            data: Training data as Pandas or Dask DataFrame
            **kwargs: Additional training parameters (unused, for compatibility)
            
        Returns:
            Dictionary of training metrics (r2, rmse, mae)
            
        Raises:
            ValueError: If data validation fails
            RuntimeError: If training fails
            ValidationError: If Pydantic validation fails
        """
        logger.info("Initializing training pipeline...")

        # 0. Validate hyperparameters using enhanced validator
        try:
            hyperparam_config = self.config.to_hyperparameter_config()
            logger.debug(f"Hyperparameters validated: n_estimators={hyperparam_config.n_estimators}, "
                       f"test_size={hyperparam_config.test_size}")
        except Exception as e:
            logger.warning(f"Hyperparameter validation warning: {e}")

        # 1. Validate training data using ModelBase validation
        required_cols = self.config.features + [self.config.target]
        clean_df = self._validate_training_data(
            data=data,
            required_columns=required_cols,
            min_rows=10  # Minimum rows for meaningful train/test split
        )

        # 2. Extract features and target
        x = clean_df[self.config.features]
        y = clean_df[self.config.target]

        # 3. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        # 4. Define Pipeline (Scaler + Model)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1
            ))
        ])

        # 5. Fit Model
        logger.info(f"Fitting model on {len(X_train)} samples...")
        self.pipeline.fit(X_train, y_train)

        # 6. Evaluate
        preds = self.pipeline.predict(X_test)
        metrics_dict = {
            'r2': r2_score(y_test, preds),
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'mae': mean_absolute_error(y_test, preds)
        }
        
        # 7. Validate metrics using Pydantic
        try:
            self.metrics = ModelMetrics(metrics=metrics_dict)
        except ValidationError as e:
            logger.error(f"Invalid metrics: {e}")
            raise
        
        # 7b. Check performance using enhanced validator
        try:
            is_good = self.metrics.is_good_performance(threshold=0.7)
            if is_good:
                logger.info("Model performance meets quality threshold (>= 0.7)")
            else:
                logger.warning("Model performance below quality threshold (< 0.7)")
        except Exception as e:
            logger.debug(f"Could not evaluate performance quality: {e}")
        
        # 8. Mark model as trained
        self._set_trained(True)
        
        logger.info(f"Training success. Metrics: {self.metrics.metrics}")
        return self.metrics.metrics

    def predict(
        self,
        data: Union[pd.DataFrame, 'dd.DataFrame'],
        **kwargs
    ) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """
        Make predictions on new data with Pydantic validation.
        
        Args:
            data: Prediction data as Pandas or Dask DataFrame
            **kwargs: Additional prediction parameters (unused, for compatibility)
            
        Returns:
            Predictions as numpy array
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If data validation fails
        """
        if not self.pipeline or not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        
        # Validate prediction data using ModelBase validation
        clean_df = self._validate_prediction_data(
            data=data,
            required_columns=self.config.features,
            min_rows=1
        )
        
        # Extract features
        x = clean_df[self.config.features]
        
        # Make predictions
        predictions = self.pipeline.predict(x)
        logger.debug(f"Made predictions for {len(x)} samples")
        return predictions
    
    def save(self, filepath: Union[str, Path] = None) -> None:
        """
        Save the model pipeline to disk.
        
        Args:
            filepath: Optional path to save the model. If None, uses config.model_path
            
        Raises:
            RuntimeError: If pipeline is not trained
            IOError: If saving fails
        """
        if not self.pipeline or not self.is_trained:
            raise RuntimeError("No trained pipeline to save.")
        
        save_path = Path(filepath) if filepath else self.config.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(self.pipeline, save_path)
            logger.info(f"Pipeline saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise IOError(f"Failed to save pipeline to {save_path}: {e}") from e
    
    def load(self, filepath: Union[str, Path] = None) -> None:
        """
        Load the model pipeline from disk.
        
        Args:
            filepath: Optional path to load the model from. If None, uses config.model_path
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            IOError: If loading fails
        """
        load_path = Path(filepath) if filepath else self.config.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        try:
            self.pipeline = joblib.load(load_path)
            self._set_trained(True)
            logger.info(f"Pipeline loaded from {load_path}")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise IOError(f"Failed to load pipeline from {load_path}: {e}") from e