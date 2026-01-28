"""
Model Utilities Module
Purpose: Central configuration and utilities for ML models with Pydantic validation.
"""

import logging
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Import enhanced Pydantic validators
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pydantic_enhancements import ModelPathConfig, ModelHyperparameterConfig


class ModelConfig(BaseModel):
    """
    Central Configuration for the model with Pydantic validation.
    
    Provides type-safe configuration with automatic validation for:
    - Paths and directories
    - Feature and target columns
    - Model hyperparameters
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Paths
    artifacts_dir: Path = Field(
        default=Path("artifacts"),
        description="Directory for storing model artifacts"
    )
    model_filename: str = Field(
        default="spills_pipeline.joblib",
        description="Filename for the saved model"
    )
    plots_dir: Path = Field(
        default=Path("plots"),
        description="Directory for storing plots and visualizations"
    )

    # Data Settings
    features: List[str] = Field(
        default_factory=lambda: [
            'Avg_Annual_Spills', 'Latitude', 'Longitude'
        ],
        description="List of feature column names to use for training"
    )
    target: str = Field(
        default='Predicted Annual Spill Frequence Post Scheme',
        description="Target column name for prediction"
    )

    # Model Settings
    n_estimators: int = Field(
        default=100,
        gt=0,
        description="Number of estimators for ensemble models (must be > 0)"
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility (must be >= 0)"
    )
    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Proportion of data to use for testing (must be between 0 and 1)"
    )

    @field_validator('model_filename')
    @classmethod
    def validate_model_filename(cls, v: str) -> str:
        """Validate model filename is not empty."""
        if not v or not v.strip():
            raise ValueError("model_filename cannot be empty")
        return v.strip()

    @field_validator('target')
    @classmethod
    def validate_target(cls, v: str) -> str:
        """Validate target column name is not empty."""
        if not v or not v.strip():
            raise ValueError("target cannot be empty")
        return v.strip()

    @field_validator('features')
    @classmethod
    def validate_features(cls, v: List[str]) -> List[str]:
        """Validate features list is not empty and contains no duplicates."""
        if not v:
            raise ValueError("features list cannot be empty")
        if len(v) != len(set(v)):
            raise ValueError("features list contains duplicates")
        return [f.strip() for f in v if f.strip()]

    def model_post_init(self, __context) -> None:
        """Create directories if they don't exist after model initialization."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def model_path(self) -> Path:
        """Full path to the saved model file."""
        return self.artifacts_dir / self.model_filename
    
    @property
    def plots_path(self) -> Path:
        """Full path to the plots directory."""
        return self.plots_dir
    
    def to_path_config(self) -> ModelPathConfig:
        """
        Convert to ModelPathConfig for enhanced path validation.
        
        Returns:
            ModelPathConfig instance
        """
        return ModelPathConfig(
            artifacts_dir=self.artifacts_dir,
            model_filename=self.model_filename,
            plots_dir=self.plots_dir
        )
    
    def to_hyperparameter_config(self) -> ModelHyperparameterConfig:
        """
        Convert to ModelHyperparameterConfig for enhanced hyperparameter validation.
        
        Returns:
            ModelHyperparameterConfig instance
        """
        return ModelHyperparameterConfig(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            test_size=self.test_size
        )


def get_logger(name: str) -> logging.Logger:
    """
    Returns a standardized logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)