"""
Model Comparison Module
Purpose: Compare multiple ML algorithms and select the best performing model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

from .model_utils import ModelConfig, get_logger
from .model_base import ModelBase, ModelMetrics

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

logger = get_logger(__name__)


class ModelComparator:
    """
    Compare multiple ML models and select the best performing one.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model comparator.
        
        Args:
            config: ModelConfig instance
        """
        self.config = config
        self.models_ = {}
        self.results_ = {}
        self.best_model_ = None
        self.best_model_name_ = None
    
    def _create_model(self, model_name: str) -> Pipeline:
        """Create a model pipeline by name."""
        scaler = StandardScaler()
        
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state
            ),
            'adaboost': AdaBoostRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state
            ),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.config.random_state),
            'lasso': Lasso(random_state=self.config.random_state),
            'decision_tree': DecisionTreeRegressor(
                random_state=self.config.random_state
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        return Pipeline([
            ('scaler', scaler),
            ('regressor', models[model_name])
        ])
    
    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_names: Optional[List[str]] = None,
        cv: int = 5,
        scoring: str = 'r2'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            X: Feature data
            y: Target data
            model_names: List of model names to compare. If None, compares all available.
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with comparison results for each model
        """
        if model_names is None:
            model_names = [
                'random_forest',
                'gradient_boosting',
                'adaboost',
                'linear_regression',
                'ridge',
                'lasso',
                'decision_tree'
            ]
        
        logger.info(f"Comparing {len(model_names)} models using {cv}-fold CV...")
        
        results = {}
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.config.random_state)
        
        for model_name in model_names:
            try:
                logger.info(f"Evaluating {model_name}...")
                model = self._create_model(model_name)
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=kfold,
                    scoring=scoring,
                    n_jobs=-1
                )
                
                # Fit and evaluate on full data
                model.fit(X, y)
                y_pred = model.predict(X)
                
                results[model_name] = {
                    'cv_mean': float(np.mean(cv_scores)),
                    'cv_std': float(np.std(cv_scores)),
                    'cv_scores': cv_scores.tolist(),
                    'r2': float(r2_score(y, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                    'mae': float(mean_absolute_error(y, y_pred)),
                    'model': model
                }
                
                logger.info(f"{model_name}: CV {scoring} = {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.results_ = results
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            self.best_model_name_ = max(
                valid_results.keys(),
                key=lambda x: valid_results[x]['cv_mean']
            )
            self.best_model_ = valid_results[self.best_model_name_]['model']
            logger.info(f"Best model: {self.best_model_name_} (CV score: {valid_results[self.best_model_name_]['cv_mean']:.4f})")
        
        return results
    
    def get_best_model(self) -> Tuple[str, Pipeline]:
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model_name, model_pipeline)
        """
        if self.best_model_ is None:
            raise ValueError("No comparison performed yet. Call compare_models() first.")
        return self.best_model_name_, self.best_model_
    
    def create_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_names: Optional[List[str]] = None,
        method: str = 'voting'
    ) -> Pipeline:
        """
        Create an ensemble model from multiple models.
        
        Args:
            X: Feature data
            y: Target data
            model_names: List of model names to include in ensemble
            method: Ensemble method ('voting' or 'stacking')
            
        Returns:
            Ensemble model pipeline
        """
        if model_names is None:
            model_names = ['random_forest', 'gradient_boosting', 'adaboost']
        
        logger.info(f"Creating {method} ensemble from {len(model_names)} models...")
        
        if method == 'voting':
            from sklearn.ensemble import VotingRegressor
            
            estimators = [
                (name, self._create_model(name).named_steps['regressor'])
                for name in model_names
            ]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_scaled, y)
            
            # Wrap in pipeline
            ensemble_pipeline = Pipeline([
                ('scaler', scaler),
                ('ensemble', ensemble)
            ])
            
            logger.info("Ensemble model created successfully")
            return ensemble_pipeline
        
        elif method == 'stacking':
            try:
                from sklearn.ensemble import StackingRegressor
                
                estimators = [
                    (name, self._create_model(name).named_steps['regressor'])
                    for name in model_names
                ]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                ensemble = StackingRegressor(
                    estimators=estimators,
                    final_estimator=RandomForestRegressor(
                        n_estimators=50,
                        random_state=self.config.random_state
                    ),
                    cv=5
                )
                ensemble.fit(X_scaled, y)
                
                # Wrap in pipeline
                ensemble_pipeline = Pipeline([
                    ('scaler', scaler),
                    ('ensemble', ensemble)
                ])
                
                logger.info("Stacking ensemble model created successfully")
                return ensemble_pipeline
            
            except ImportError:
                logger.warning("Stacking requires scikit-learn >= 0.22. Using voting instead.")
                return self.create_ensemble(X, y, model_names, method='voting')
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}. Use 'voting' or 'stacking'")

