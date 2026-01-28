"""
Advanced Feature Engineering Module
Purpose: Automated feature creation, selection, and importance analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, cast
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE,
    SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

from .model_utils import get_logger

logger = get_logger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering with automated creation and selection.
    """
    
    def __init__(self):
        """Initialize advanced feature engineer."""
        self.feature_importance_ = None
        self.selected_features_ = None
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features from specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from. If None, uses all numeric columns.
            degree: Degree of polynomial features
            include_bias: Whether to include bias term
            
        Returns:
            DataFrame with original and polynomial features
        """
        logger.info(f"Creating polynomial features (degree={degree})...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[columns].values
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Create DataFrame
        df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
        
        # Combine with original DataFrame
        result_df = pd.concat([df, df_poly], axis=1)
        
        logger.info(f"Created {len(feature_names)} polynomial features")
        return result_df
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create interaction features (multiplication of pairs).
        
        Args:
            df: Input DataFrame
            columns: Columns to create interactions from. If None, uses all numeric columns.
            
        Returns:
            DataFrame with interaction features added
        """
        logger.info("Creating interaction features...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Type narrowing: ensure columns is List[str] for type checker
        columns_list: List[str] = columns if columns is not None else []
        
        result_df = df.copy()
        
        # Create pairwise interactions
        for i, col1 in enumerate(columns_list):
            for col2 in columns_list[i+1:]:
                interaction_name = f"{col1}_x_{col2}"
                result_df[interaction_name] = df[col1] * df[col2]
        
        logger.info(f"Created {len(columns_list) * (len(columns_list) - 1) // 2} interaction features")
        return result_df
    
    def create_statistical_features(
        self,
        df: pd.DataFrame,
        group_by: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create statistical features (mean, std, min, max) for groups.
        
        Args:
            df: Input DataFrame
            group_by: Column to group by (e.g., 'Water company')
            columns: Columns to calculate statistics for. If None, uses all numeric columns.
            
        Returns:
            DataFrame with statistical features added
        """
        logger.info("Creating statistical features...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Type narrowing: ensure columns is List[str] for type checker
        columns_list: List[str] = columns if columns is not None else []
        
        result_df = df.copy()
        
        if group_by and group_by in df.columns:
            for col in columns_list:
                grouped = df.groupby(group_by)[col]
                mean_series = cast(pd.Series, grouped.mean())
                std_series = cast(pd.Series, grouped.std())
                min_series = cast(pd.Series, grouped.min())
                max_series = cast(pd.Series, grouped.max())
                result_df[f"{col}_group_mean"] = df[group_by].map(mean_series.to_dict())  # type: ignore[arg-type]
                result_df[f"{col}_group_std"] = df[group_by].map(std_series.to_dict())  # type: ignore[arg-type]
                result_df[f"{col}_group_min"] = df[group_by].map(min_series.to_dict())  # type: ignore[arg-type]
                result_df[f"{col}_group_max"] = df[group_by].map(max_series.to_dict())  # type: ignore[arg-type]
            
            logger.info(f"Created statistical features grouped by {group_by}")
        else:
            # Global statistics
            for col in columns_list:
                result_df[f"{col}_mean"] = df[col].mean()
                result_df[f"{col}_std"] = df[col].std()
        
        return result_df
    
    def select_features_univariate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        score_func: str = 'f_regression'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using univariate statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            score_func: Scoring function ('f_regression' or 'mutual_info')
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        logger.info(f"Selecting top {k} features using {score_func}...")
        
        if score_func == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif score_func == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"Unknown score_func: {score_func}")
        
        X_selected = selector.fit_transform(X, y)
        selected_features: List[str] = X.columns[selector.get_support()].tolist()
        
        self.selected_features_ = selected_features
        
        logger.info(f"Selected features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features  # type: ignore[arg-type]
    
    def select_features_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        estimator=None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            estimator: Base estimator. If None, uses RandomForestRegressor
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        logger.info(f"Selecting {n_features} features using RFE...")
        
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features: List[str] = X.columns[selector.get_support()].tolist()
        
        self.selected_features_ = selected_features
        
        logger.info(f"Selected features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features  # type: ignore[arg-type]
    
    def select_features_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        estimator=None
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """
        Select features based on importance from tree-based model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            threshold: Minimum importance threshold
            max_features: Maximum number of features to select
            estimator: Base estimator. If None, uses RandomForestRegressor
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names, feature_importance_dict)
        """
        logger.info("Selecting features based on importance...")
        
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        estimator.fit(X, y)
        
        # Get feature importance
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        else:
            # For linear models, use coefficients
            importances = np.abs(estimator.coef_)  # type: ignore[attr-defined]
        
        feature_importance: Dict[str, float] = {str(k): float(v) for k, v in zip(X.columns, importances)}
        self.feature_importance_ = feature_importance
        
        # Select features
        if threshold is not None:
            selector = SelectFromModel(estimator, threshold=threshold)
            selector.fit(X, y)
        elif max_features is not None:
            # Select top max_features
            top_indices = np.argsort(importances)[-max_features:]
            threshold_value = importances[top_indices[0]]
            selector = SelectFromModel(estimator, threshold=threshold_value)
            selector.fit(X, y)
        else:
            # Use median importance as threshold
            selector = SelectFromModel(estimator, threshold='median')
            selector.fit(X, y)
        
        X_selected = selector.transform(X)
        selected_features: List[str] = X.columns[selector.get_support()].tolist()
        
        self.selected_features_ = selected_features
        
        logger.info(f"Selected {len(selected_features)} features based on importance")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features, feature_importance  # type: ignore[arg-type]
    
    def apply_pca(
        self,
        X: pd.DataFrame,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, PCA]:
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Args:
            X: Feature DataFrame
            n_components: Number of components. If None, selects to explain variance_threshold variance.
            variance_threshold: Minimum variance to explain (if n_components is None)
            
        Returns:
            Tuple of (transformed_df, pca_model)
        """
        logger.info("Applying PCA...")
        
        if n_components is None:
            pca = PCA()
            pca.fit(X)
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = int(np.argmax(cumsum_variance >= variance_threshold) + 1)
            logger.info(f"Selected {n_components} components to explain {variance_threshold*100}% variance")
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        component_names: List[str] = [f"PC{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=component_names, index=X.index)  # type: ignore[arg-type]
        
        logger.info(f"PCA completed. Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        return df_pca, pca
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        return self.feature_importance_  # type: ignore[return-value]
    
    def get_selected_features(self) -> Optional[List[str]]:
        """Get list of selected features."""
        return self.selected_features_

