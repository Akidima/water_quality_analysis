"""
Hyperparameter Tuning Module
Purpose: Grid search, random search, and cross-validation for model optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    KFold,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
import joblib

from .model_utils import ModelConfig, get_logger
from .model_base import ModelBase

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning with grid search, random search, and cross-validation.
    """
    
    def __init__(self, config: ModelConfig, model_type: str = 'random_forest'):
        """
        Initialize hyperparameter tuner.
        
        Args:
            config: ModelConfig instance
            model_type: Type of model ('random_forest', 'gradient_boosting', etc.)
        """
        self.config = config
        self.model_type = model_type
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.best_estimator_ = None
    
    def _create_base_pipeline(self) -> Pipeline:
        """Create base pipeline for the model."""
        if self.model_type == 'random_forest':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(random_state=self.config.random_state))
            ])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _get_default_param_grid(self) -> Dict[str, List[Any]]:
        """Get default parameter grid for the model type."""
        if self.model_type == 'random_forest':
            return {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            }
        else:
            return {}
    
    def grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 5,
        scoring: str = 'r2',
        n_jobs: int = -1,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter optimization.
        
        Args:
            X: Feature data
            y: Target data
            param_grid: Parameter grid to search. If None, uses default.
            cv: Number of cross-validation folds
            scoring: Scoring metric ('r2', 'neg_mean_squared_error', etc.)
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            
        Returns:
            Dictionary with best parameters, score, and results
        """
        logger.info(f"Starting grid search with {cv}-fold cross-validation...")
        
        pipeline = self._create_base_pipeline()
        param_grid = param_grid or self._get_default_param_grid()
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.cv_results_ = grid_search.cv_results_
        self.best_estimator_ = grid_search.best_estimator_
        
        logger.info(f"Grid search completed. Best score: {self.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params_}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': self.cv_results_,
            'best_estimator': self.best_estimator_
        }
    
    def random_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_distributions: Optional[Dict[str, List[Any]]] = None,
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = 'r2',
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform random search for hyperparameter optimization.
        
        Args:
            X: Feature data
            y: Target data
            param_distributions: Parameter distributions to sample from
            n_iter: Number of parameter settings sampled
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with best parameters, score, and results
        """
        logger.info(f"Starting random search with {n_iter} iterations...")
        
        pipeline = self._create_base_pipeline()
        param_distributions = param_distributions or self._get_default_param_grid()
        random_state = random_state or self.config.random_state
        
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        self.best_params_ = random_search.best_params_
        self.best_score_ = random_search.best_score_
        self.cv_results_ = random_search.cv_results_
        self.best_estimator_ = random_search.best_estimator_
        
        logger.info(f"Random search completed. Best score: {self.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params_}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': self.cv_results_,
            'best_estimator': self.best_estimator_
        }
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: Optional[List[str]] = None,
        return_train_score: bool = False
    ) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            X: Feature data
            y: Target data
            cv: Number of folds
            scoring: List of scoring metrics. If None, uses ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            return_train_score: Whether to return training scores
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting {cv}-fold cross-validation...")
        
        pipeline = self._create_base_pipeline()
        
        if scoring is None:
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
        cv_results = {}
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.config.random_state)
        
        for metric in scoring:
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=kfold,
                scoring=metric,
                n_jobs=-1
            )
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
            logger.info(f"{metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        return cv_results
    
    def get_best_estimator(self):
        """Get the best estimator from tuning."""
        if self.best_estimator_ is None:
            raise ValueError("No tuning performed yet. Call grid_search() or random_search() first.")
        return self.best_estimator_


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning (requires scikit-optimize).
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Bayesian optimizer.
        
        Args:
            config: ModelConfig instance
        """
        self.config = config
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
            self.gp_minimize = gp_minimize
            self.Real = Real
            self.Integer = Integer
            self.use_named_args = use_named_args
            self.SKOPT_AVAILABLE = True
        except ImportError:
            self.SKOPT_AVAILABLE = False
            logger.warning("scikit-optimize not available. Install with: pip install scikit-optimize")
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        n_calls: int = 50,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization.
        
        Args:
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds
            n_calls: Number of iterations
            random_state: Random state
            
        Returns:
            Dictionary with optimization results
        """
        if not self.SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        
        logger.info(f"Starting Bayesian optimization with {n_calls} iterations...")
        
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        
        # Define search space
        space = [
            Integer(50, 300, name='n_estimators'),
            Integer(5, 50, name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 10, name='min_samples_leaf')
        ]
        
        # Create pipeline function
        @self.use_named_args(space=space)
        def objective(**params):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'] if params['max_depth'] > 0 else None,
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    random_state=self.config.random_state,
                    n_jobs=-1
                ))
            ])
            
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            return -np.mean(scores)  # Minimize (negate because we want to minimize MSE)
        
        # Run optimization
        result = self.gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=random_state or self.config.random_state,
            verbose=True
        )
        
        best_params = {
            'n_estimators': result.x[0],
            'max_depth': result.x[1] if result.x[1] > 0 else None,
            'min_samples_split': result.x[2],
            'min_samples_leaf': result.x[3]
        }
        
        logger.info(f"Bayesian optimization completed. Best score: {-result.fun:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': -result.fun,
            'optimization_result': result
        }

