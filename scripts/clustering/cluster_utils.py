"""
Water Quality Site Clustering Utilities
=======================================
Reusable clustering functions for water quality monitoring sites.

This module provides production-ready clustering functionality with:
- Pydantic-enhanced configuration and validation
- K-Means clustering with automatic optimal k detection
- Comprehensive cluster quality metrics
- Visualization and reporting capabilities

Example usage:
    from scripts.clustering import SiteClusterer, ClusteringConfig
    
    config = ClusteringConfig(min_clusters=3, max_clusters=8)
    clusterer = SiteClusterer(config)
    success = clusterer.fit(df)
    clusterer.create_visualizations()
"""

import os
import joblib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, ClassVar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    ValidationError
)

# Optional dependency for automatic elbow detection
try:
    from kneed import KneeLocator  # type: ignore[import-not-found]
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False
    KneeLocator = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Configuration Models
# =============================================================================

class ClusteringConfig(BaseModel):
    """
    Pydantic-validated configuration for clustering analysis.
    
    This provides type-safe configuration with automatic validation
    of clustering parameters, ensuring valid ranges and logical consistency.
    
    Attributes:
        output_dir: Directory for saving plots and visualizations
        model_save_path: Path for saving the clustering model
        scaler_save_path: Path for saving the feature scaler
        min_clusters: Minimum number of clusters to try (must be >= 2)
        max_clusters: Maximum number of clusters to try
        random_state: Random seed for reproducibility
        n_init: Number of K-Means initializations
        max_iter: Maximum iterations per K-Means run
        base_features: Required feature columns for clustering
        optional_features: Optional feature columns to include if available
        figure_size: Figure size for visualizations (width, height)
        dpi: DPI for saved figures
    
    Example:
        >>> config = ClusteringConfig(
        ...     min_clusters=3,
        ...     max_clusters=10,
        ...     output_dir='clustering_results'
        ... )
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        extra='forbid'
    )
    
    # File paths
    output_dir: str = Field(
        default='plots',
        description="Directory for saving output plots"
    )
    model_save_path: str = Field(
        default='models/clustering_model.pkl',
        description="Path for saving the clustering model"
    )
    scaler_save_path: str = Field(
        default='models/clustering_scaler.pkl',
        description="Path for saving the feature scaler"
    )
    
    # Clustering parameters
    min_clusters: int = Field(
        default=2,
        ge=2,
        description="Minimum number of clusters (must be >= 2)"
    )
    max_clusters: int = Field(
        default=8,
        ge=2,
        le=50,
        description="Maximum number of clusters (2-50)"
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    n_init: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of K-Means initializations (1-100)"
    )
    max_iter: int = Field(
        default=300,
        ge=100,
        le=10000,
        description="Maximum iterations per K-Means run (100-10000)"
    )
    
    # Feature selection
    base_features: List[str] = Field(
        default_factory=lambda: [
            'Latitude',
            'Longitude',
            'Avg_Annual_Spills'
        ],
        min_length=1,
        description="Required feature columns for clustering"
    )
    
    optional_features: List[str] = Field(
        default_factory=lambda: [
            'Spill_Trend',
            'Predicted Annual Spill Frequence Post Scheme',
            'Ecological High Priority Site Flag',
            'Non-bathing Priority Site Flag',
            'Bathing Water Discharge Flag',
            'Shellfish Water Discharge Flag'
        ],
        description="Optional feature columns to include if available"
    )
    
    # Visualization
    figure_size: Tuple[int, int] = Field(
        default=(16, 12),
        description="Figure size for visualizations (width, height)"
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="DPI for saved figures (72-600)"
    )
    
    @field_validator('base_features', 'optional_features')
    @classmethod
    def validate_feature_lists(cls, v: List[str]) -> List[str]:
        """Validate that feature lists contain non-empty strings."""
        if not v:
            return v
        cleaned = [f.strip() for f in v if f and f.strip()]
        if len(cleaned) != len(v):
            logger.warning("Some empty feature names were removed from the list")
        return cleaned
    
    @field_validator('figure_size')
    @classmethod
    def validate_figure_size(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Validate figure size dimensions."""
        if len(v) != 2:
            raise ValueError("figure_size must be a tuple of (width, height)")
        width, height = v
        if width < 4 or width > 40:
            raise ValueError(f"Figure width must be between 4 and 40, got {width}")
        if height < 4 or height > 40:
            raise ValueError(f"Figure height must be between 4 and 40, got {height}")
        return v
    
    @model_validator(mode='after')
    def validate_cluster_range(self) -> 'ClusteringConfig':
        """Validate that min_clusters <= max_clusters."""
        if self.min_clusters > self.max_clusters:
            raise ValueError(
                f"min_clusters ({self.min_clusters}) cannot be greater than "
                f"max_clusters ({self.max_clusters})"
            )
        return self
    
    def create_directories(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directories: {self.output_dir}, {Path(self.model_save_path).parent}")


class ClusterMetrics(BaseModel):
    """
    Pydantic model for cluster quality metrics.
    
    Provides validated storage for clustering evaluation metrics
    with automatic quality interpretation.
    
    Attributes:
        silhouette_score: Measure of cluster separation (-1 to 1, higher is better)
        davies_bouldin_score: Measure of cluster similarity (0 to inf, lower is better)
        calinski_harabasz_score: Ratio of between/within cluster variance (higher is better)
        n_clusters: Number of clusters
        inertia: Within-cluster sum of squares
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    silhouette_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Silhouette score (-1 to 1, higher is better)"
    )
    davies_bouldin_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Davies-Bouldin score (0 to inf, lower is better)"
    )
    calinski_harabasz_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Calinski-Harabasz score (higher is better)"
    )
    n_clusters: int = Field(
        default=0,
        ge=0,
        description="Number of clusters"
    )
    inertia: float = Field(
        default=0.0,
        ge=0.0,
        description="Within-cluster sum of squares"
    )
    
    def get_quality_interpretation(self) -> str:
        """
        Interpret clustering quality based on silhouette score.
        
        Returns:
            Human-readable quality assessment string
        """
        silhouette = self.silhouette_score
        
        if silhouette > 0.7:
            return "Excellent - Clusters are very distinct and well-separated"
        elif silhouette > 0.5:
            return "Good - Clusters are reasonably well-defined"
        elif silhouette > 0.3:
            return "Fair - Clusters have some overlap"
        elif silhouette > 0.0:
            return "Poor - Clusters are not well-defined"
        else:
            return "Very Poor - Clusters may be overlapping significantly"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary with interpretation."""
        return {
            'silhouette_score': self.silhouette_score,
            'davies_bouldin_score': self.davies_bouldin_score,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            'n_clusters': self.n_clusters,
            'inertia': self.inertia,
            'quality_interpretation': self.get_quality_interpretation()
        }


class ClusterAnalysisResult(BaseModel):
    """
    Pydantic model for comprehensive cluster analysis results.
    
    Stores all results from a clustering analysis including
    cluster assignments, statistics, and quality metrics.
    
    Attributes:
        optimal_k: Optimal number of clusters found
        cluster_sizes: Number of sites in each cluster
        cluster_stats: Statistical summary per cluster
        worst_cluster: ID of the worst-performing cluster
        metrics: Cluster quality metrics
        feature_names: List of features used for clustering
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    optimal_k: int = Field(
        ...,
        ge=1,
        description="Optimal number of clusters"
    )
    cluster_sizes: Dict[int, int] = Field(
        default_factory=dict,
        description="Number of sites in each cluster"
    )
    cluster_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Statistical summary per cluster"
    )
    avg_spills_per_cluster: Dict[int, float] = Field(
        default_factory=dict,
        description="Average spills per cluster"
    )
    worst_cluster: Optional[int] = Field(
        default=None,
        description="ID of the worst-performing cluster"
    )
    metrics: ClusterMetrics = Field(
        default_factory=ClusterMetrics,
        description="Cluster quality metrics"
    )
    feature_names: List[str] = Field(
        default_factory=list,
        description="Features used for clustering"
    )
    company_distribution: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Company distribution across clusters"
    )
    priority_analysis: Dict[str, Dict[int, int]] = Field(
        default_factory=dict,
        description="Priority site counts per cluster"
    )


# =============================================================================
# Clustering Validation Utilities
# =============================================================================

class ClusterValidator:
    """
    Validates clustering results to ensure quality.
    
    Provides static methods for calculating and interpreting
    cluster quality metrics.
    
    Example:
        >>> metrics = ClusterValidator.calculate_cluster_metrics(X_scaled, labels)
        >>> interpretation = ClusterValidator.interpret_metrics(metrics)
    """
    
    @staticmethod
    def calculate_cluster_metrics(
        X: np.ndarray,
        labels: np.ndarray
    ) -> ClusterMetrics:
        """
        Calculate clustering quality metrics.
        
        Args:
            X: Scaled feature data
            labels: Cluster assignments
            
        Returns:
            ClusterMetrics object with quality scores
        """
        n_clusters = len(np.unique(labels))
        
        # Skip metrics if only 1 cluster
        if n_clusters < 2:
            return ClusterMetrics(n_clusters=n_clusters)
        
        try:
            silhouette = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            
            return ClusterMetrics(
                silhouette_score=float(silhouette),
                davies_bouldin_score=float(db_score),
                calinski_harabasz_score=float(ch_score),
                n_clusters=n_clusters
            )
        except Exception as e:
            logger.warning(f"Error calculating cluster metrics: {e}")
            return ClusterMetrics(n_clusters=n_clusters)
    
    @staticmethod
    def interpret_metrics(metrics: Union[ClusterMetrics, Dict[str, float]]) -> str:
        """
        Interpret clustering quality metrics.
        
        Args:
            metrics: ClusterMetrics object or dictionary with silhouette_score
            
        Returns:
            Human-readable quality interpretation
        """
        if isinstance(metrics, ClusterMetrics):
            return metrics.get_quality_interpretation()
        
        silhouette = metrics.get('silhouette_score', 0.0)
        
        if silhouette > 0.7:
            return "Excellent - Clusters are very distinct and well-separated"
        elif silhouette > 0.5:
            return "Good - Clusters are reasonably well-defined"
        elif silhouette > 0.3:
            return "Fair - Clusters have some overlap"
        else:
            return "Poor - Clusters are not well-defined"
    
    @staticmethod
    def validate_data_for_clustering(
        df: pd.DataFrame,
        required_features: List[str],
        min_samples: int = 10
    ) -> Tuple[bool, List[str]]:
        """
        Validate that data is suitable for clustering.
        
        Args:
            df: Input DataFrame
            required_features: List of required feature columns
            min_samples: Minimum number of samples required
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check minimum samples
        if len(df) < min_samples:
            issues.append(f"Insufficient samples: {len(df)} < {min_samples}")
        
        # Check required features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            issues.append(f"Missing required features: {missing_features}")
        
        # Check for all-NaN columns in required features
        for feature in required_features:
            if feature in df.columns and bool(df[feature].isna().all()):
                issues.append(f"Feature '{feature}' has all NaN values")
        
        # Check for constant columns
        for feature in required_features:
            if feature in df.columns:
                if df[feature].nunique() <= 1:
                    issues.append(f"Feature '{feature}' has no variance (constant)")
        
        return len(issues) == 0, issues


# =============================================================================
# Optimal Cluster Finder
# =============================================================================

class OptimalClusterFinder:
    """
    Finds the optimal number of clusters using the elbow method.
    
    Uses K-Means inertia values and the KneeLocator algorithm
    to automatically detect the optimal number of clusters.
    
    Attributes:
        config: ClusteringConfig object
        inertias: List of inertia values for each k
        silhouette_scores: List of silhouette scores for each k
    
    Example:
        >>> finder = OptimalClusterFinder(config)
        >>> optimal_k, inertias = finder.find_optimal_k(X_scaled)
    """
    
    def __init__(self, config: ClusteringConfig):
        """
        Initialize the optimal cluster finder.
        
        Args:
            config: ClusteringConfig object with clustering parameters
        """
        self.config = config
        self.inertias: List[float] = []
        self.silhouette_scores: List[float] = []
        self.k_range = range(config.min_clusters, config.max_clusters + 1)
    
    def find_optimal_k(
        self,
        X: np.ndarray,
        method: str = 'elbow'
    ) -> Tuple[int, List[float]]:
        """
        Find the optimal number of clusters.
        
        Args:
            X: Scaled feature data
            method: Method to use ('elbow' or 'silhouette')
            
        Returns:
            Tuple of (optimal_k, list of inertias)
        """
        logger.info(f"Finding optimal k using {method} method")
        
        self.inertias = []
        self.silhouette_scores = []
        
        # Try different numbers of clusters
        for k in self.k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.config.random_state,
                n_init=self.config.n_init,  # type: ignore[arg-type]
                max_iter=self.config.max_iter
            )
            kmeans.fit(X)
            inertia_value: float = kmeans.inertia_  # type: ignore[assignment]
            self.inertias.append(inertia_value)
            
            # Calculate silhouette score for comparison
            if k >= 2:
                labels = kmeans.labels_
                self.silhouette_scores.append(silhouette_score(X, labels))
            else:
                self.silhouette_scores.append(0.0)
        
        # Determine optimal k based on method
        if method == 'silhouette':
            optimal_k = self._find_optimal_by_silhouette()
        else:
            optimal_k = self._find_optimal_by_elbow()
        
        logger.info(f"Optimal k found: {optimal_k}")
        return optimal_k, self.inertias
    
    def _find_optimal_by_elbow(self) -> int:
        """Find optimal k using elbow method."""
        if not KNEED_AVAILABLE:
            logger.warning("kneed not available, using fallback method")
            return self._fallback_optimal_k()
        
        try:
            # KneeLocator is guaranteed to be available here due to check above
            knee_locator = KneeLocator(  # type: ignore[misc]
                list(self.k_range),
                self.inertias,
                curve='convex',
                direction='decreasing'
            )
            optimal_k = knee_locator.elbow
            
            if optimal_k is None:
                logger.warning("No clear elbow found, using fallback")
                return self._fallback_optimal_k()
            
            return optimal_k
            
        except Exception as e:
            logger.warning(f"Knee detection failed: {e}")
            return self._fallback_optimal_k()
    
    def _find_optimal_by_silhouette(self) -> int:
        """Find optimal k using maximum silhouette score."""
        if not self.silhouette_scores:
            return self._fallback_optimal_k()
        
        # Find k with maximum silhouette score
        max_idx = np.argmax(self.silhouette_scores)
        return list(self.k_range)[max_idx]
    
    def _fallback_optimal_k(self) -> int:
        """Fallback method when automatic detection fails."""
        # Use the middle of the range as a reasonable default
        return (self.config.min_clusters + self.config.max_clusters) // 2


# =============================================================================
# Main Site Clusterer Class
# =============================================================================

class SiteClusterer:
    """
    Main class for clustering water quality monitoring sites.
    
    Provides complete workflow for clustering analysis including:
    - Feature preparation and scaling
    - Automatic optimal k detection
    - K-Means clustering
    - Cluster quality evaluation
    - Visualization generation
    - Model persistence
    
    Attributes:
        config: ClusteringConfig object
        model: Fitted KMeans model
        scaler: Fitted StandardScaler
        feature_names: List of feature names used
        cluster_analysis: ClusterAnalysisResult object
        df_clustered: DataFrame with cluster assignments
    
    Example:
        >>> config = ClusteringConfig(min_clusters=3, max_clusters=8)
        >>> clusterer = SiteClusterer(config)
        >>> success = clusterer.fit(df)
        >>> if success:
        ...     clusterer.create_visualizations()
        ...     clusterer.print_summary()
        ...     clusterer.save_model()
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize the site clusterer.
        
        Args:
            config: ClusteringConfig object (uses defaults if None)
        """
        self.config = config or ClusteringConfig()
        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.cluster_analysis: Optional[ClusterAnalysisResult] = None
        self.optimal_k: Optional[int] = None
        self.df_clustered: Optional[pd.DataFrame] = None
        self.features_scaled: Optional[np.ndarray] = None
        self.inertias: List[float] = []
        self.metrics: Optional[ClusterMetrics] = None
    
    def prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare and select features for clustering.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with selected features, or None if validation fails
        """
        logger.info("Preparing features for clustering")
        
        # Validate required columns
        is_valid, issues = ClusterValidator.validate_data_for_clustering(
            df, self.config.base_features
        )
        
        if not is_valid:
            logger.error(f"Data validation failed: {issues}")
            return None
        
        # Start with base features
        selected_features = self.config.base_features.copy()
        
        # Add optional features if available
        for feature in self.config.optional_features:
            if feature in df.columns:
                selected_features.append(feature)
                logger.info(f"Added optional feature: {feature}")
        
        # Extract features - ensure DataFrame (not Series) is returned
        # Using .loc to guarantee DataFrame return type
        feature_df = df.loc[:, selected_features].copy()
        
        # Convert flag columns to integers
        flag_columns = [col for col in feature_df.columns if 'Flag' in col]
        for col in flag_columns:
            feature_df[col] = feature_df[col].astype(int)
        
        # Handle missing values with median
        feature_df = feature_df.fillna(feature_df.median())
        
        # Store feature names
        self.feature_names = feature_df.columns.tolist()
        
        logger.info(f"Selected {len(self.feature_names)} features: {self.feature_names}")
        return feature_df  # type: ignore[return-value]
    
    def fit(self, df: pd.DataFrame) -> bool:
        """
        Perform clustering on the data.
        
        Args:
            df: Input DataFrame with monitoring sites
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting clustering analysis")
        
        # Step 1: Create output directories
        self.config.create_directories()
        
        # Step 2: Prepare features
        features = self.prepare_features(df)
        if features is None:
            logger.error("Feature preparation failed")
            return False
        
        # Step 3: Scale features
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(features)
        
        # Step 4: Find optimal number of clusters
        cluster_finder = OptimalClusterFinder(self.config)
        # features_scaled is guaranteed to be non-None after assignment above
        features_scaled_for_finder: np.ndarray = self.features_scaled  # type: ignore[assignment]
        self.optimal_k, self.inertias = cluster_finder.find_optimal_k(features_scaled_for_finder)  # type: ignore[arg-type]
        
        # Step 5: Perform clustering with optimal k
        logger.info(f"Performing clustering with k={self.optimal_k}")
        self.model = KMeans(
            n_clusters=self.optimal_k,
            random_state=self.config.random_state,
            n_init=self.config.n_init,  # type: ignore[arg-type]
            max_iter=self.config.max_iter
        )
        
        cluster_labels = self.model.fit_predict(self.features_scaled)
        
        # Step 6: Evaluate cluster quality
        # features_scaled is guaranteed to be non-None after assignment above
        features_scaled_array: np.ndarray = self.features_scaled  # type: ignore[assignment]
        self.metrics = ClusterValidator.calculate_cluster_metrics(
            features_scaled_array, cluster_labels
        )
        # inertia_ is always a float after fit(), but type checker doesn't know this
        inertia_value: float = self.model.inertia_  # type: ignore[assignment]
        self.metrics.inertia = inertia_value  # type: ignore[assignment]
        
        logger.info(f"Clustering quality: {self.metrics.get_quality_interpretation()}")
        
        # Step 7: Store results
        self.df_clustered = df.copy()
        self.df_clustered['Cluster'] = cluster_labels
        
        # Step 8: Analyze clusters
        self._analyze_clusters(df)
        
        logger.info("Clustering completed successfully")
        return True
    
    def _analyze_clusters(self, df: pd.DataFrame) -> None:
        """Perform detailed analysis of clusters."""
        logger.info("Analyzing cluster characteristics")
        
        # df_clustered is guaranteed to be non-None when this method is called
        assert self.df_clustered is not None
        df_clustered: pd.DataFrame = self.df_clustered
        
        # optimal_k is guaranteed to be non-None when this method is called
        assert self.optimal_k is not None
        optimal_k: int = self.optimal_k
        
        # metrics is guaranteed to be non-None when this method is called
        assert self.metrics is not None
        metrics: ClusterMetrics = self.metrics
        
        # Basic cluster statistics
        cluster_stats = df_clustered.groupby('Cluster')['Avg_Annual_Spills'].agg([
            'mean', 'median', 'std', 'count', 'min', 'max'
        ]).round(2).to_dict()
        
        # Average spills per cluster
        avg_spills: Dict[Any, float] = df_clustered.groupby('Cluster')['Avg_Annual_Spills'].mean().to_dict()
        
        # Find worst performing cluster
        worst_cluster = max(avg_spills, key=lambda k: avg_spills[k]) if avg_spills else None
        
        # Cluster sizes
        cluster_sizes = df_clustered['Cluster'].value_counts().sort_index().to_dict()
        
        # Company distribution
        company_distribution = None
        if 'Water company' in df_clustered.columns:
            company_distribution = df_clustered.groupby(
                ['Cluster', 'Water company']
            ).size().unstack(fill_value=0).to_dict()
        
        # Priority site analysis
        priority_analysis = {}
        priority_flags = [
            'Ecological High Priority Site Flag',
            'Non-bathing Priority Site Flag',
            'Bathing Water Discharge Flag',
            'Shellfish Water Discharge Flag'
        ]
        
        for flag in priority_flags:
            if flag in df_clustered.columns:
                priority_analysis[flag] = df_clustered.groupby('Cluster')[flag].sum().to_dict()
        
        # Store comprehensive analysis
        self.cluster_analysis = ClusterAnalysisResult(
            optimal_k=optimal_k,
            cluster_sizes=cluster_sizes,
            cluster_stats=cluster_stats,
            avg_spills_per_cluster=avg_spills,
            worst_cluster=worst_cluster,
            metrics=metrics,
            feature_names=self.feature_names,
            company_distribution=company_distribution,
            priority_analysis=priority_analysis
        )
    
    def create_visualizations(self, save: bool = True) -> Figure:
        """
        Create comprehensive visualizations of clustering results.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            Matplotlib Figure object
        """
        logger.info("Creating clustering visualizations")
        
        # Assertions for required attributes
        assert self.cluster_analysis is not None, "Cluster analysis must be performed before creating visualizations"
        assert self.df_clustered is not None, "Clustered data must be available before creating visualizations"
        assert self.optimal_k is not None, "Optimal k must be determined before creating visualizations"
        
        df_clustered: pd.DataFrame = self.df_clustered
        cluster_analysis = self.cluster_analysis
        optimal_k: int = self.optimal_k
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle('Site Clustering Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Elbow Method
        k_range = range(self.config.min_clusters, self.config.max_clusters + 1)
        axes[0, 0].plot(k_range, self.inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].axvline(
            x=optimal_k,
            color='r',
            linestyle='--',
            label=f'Optimal k={optimal_k}'
        )
        axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0, 0].set_ylabel('Inertia', fontsize=12)
        axes[0, 0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cluster Size Distribution
        cluster_counts = pd.Series(cluster_analysis.cluster_sizes)
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(cluster_counts)))
        axes[0, 1].bar(
            cluster_counts.index,
            cluster_counts.values,
            color=colors,
            edgecolor='black',
            linewidth=1.5
        )
        axes[0, 1].set_title('Distribution of Sites Across Clusters', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Cluster ID', fontsize=12)
        axes[0, 1].set_ylabel('Number of Sites', fontsize=12)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add count labels
        for idx, val in cluster_counts.items():
            axes[0, 1].text(idx, val, str(val), ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Geographic Distribution
        scatter = axes[1, 0].scatter(
            df_clustered['Longitude'],
            df_clustered['Latitude'],
            c=df_clustered['Cluster'],
            cmap='viridis',
            alpha=0.6,
            s=20,
            edgecolors='black',
            linewidth=0.5
        )
        axes[1, 0].set_xlabel('Longitude', fontsize=12)
        axes[1, 0].set_ylabel('Latitude', fontsize=12)
        axes[1, 0].set_title('Geographic Distribution of Clusters', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=axes[1, 0])
        cbar.set_label('Cluster ID', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Spill Rate Distribution by Cluster
        cluster_spill_data = []
        cluster_labels = []
        for cluster in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[
                df_clustered['Cluster'] == cluster
            ]['Avg_Annual_Spills']
            # Ensure cluster_data is a Series before accessing .values
            if isinstance(cluster_data, pd.Series):
                cluster_spill_data.append(cluster_data.values)
            else:
                cluster_spill_data.append(np.array(cluster_data))
            cluster_labels.append(f'C{cluster}')
        
        bp = axes[1, 1].boxplot(
            cluster_spill_data,
            labels=cluster_labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
        )
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 1].set_title('Spill Rate Distribution by Cluster', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Cluster', fontsize=12)
        axes[1, 1].set_ylabel('Average Annual Spills', fontsize=12)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Highlight worst cluster
        if cluster_analysis.worst_cluster is not None:
            sorted_clusters = sorted(df_clustered['Cluster'].unique())
            if cluster_analysis.worst_cluster in sorted_clusters:
                worst_idx = sorted_clusters.index(cluster_analysis.worst_cluster)
                axes[1, 1].get_xticklabels()[worst_idx].set_color('red')
                axes[1, 1].get_xticklabels()[worst_idx].set_weight('bold')
        
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.config.output_dir, 'clustering_analysis.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=self.config.dpi)
            logger.info(f"Visualizations saved to {output_path}")
        
        return fig
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of clustering results."""
        # Assertions for required attributes
        assert self.cluster_analysis is not None, "Cluster analysis must be performed before printing summary"
        assert self.metrics is not None, "Metrics must be calculated before printing summary"
        assert self.optimal_k is not None, "Optimal k must be determined before printing summary"
        
        cluster_analysis = self.cluster_analysis
        metrics = self.metrics
        optimal_k: int = self.optimal_k
        
        print("\n" + "=" * 70)
        print("CLUSTERING ANALYSIS SUMMARY")
        print("=" * 70)
        
        print(f"\nClustering Configuration:")
        print(f"   Optimal number of clusters: {optimal_k}")
        print(f"   Features used: {len(self.feature_names)}")
        print(f"   Feature list: {', '.join(self.feature_names)}")
        
        print(f"\nQuality Metrics:")
        print(f"   Silhouette Score: {metrics.silhouette_score:.3f}")
        print(f"   Davies-Bouldin Score: {metrics.davies_bouldin_score:.3f}")
        print(f"   Calinski-Harabasz Score: {metrics.calinski_harabasz_score:.1f}")
        print(f"   Quality Assessment: {metrics.get_quality_interpretation()}")
        
        print(f"\nCluster Sizes:")
        for cluster_id, size in cluster_analysis.cluster_sizes.items():
            print(f"   Cluster {cluster_id}: {size} sites")
        
        print(f"\nHigh Risk Analysis:")
        if cluster_analysis.worst_cluster is not None:
            worst = cluster_analysis.worst_cluster
            avg_spills = cluster_analysis.avg_spills_per_cluster.get(worst, 0)
            print(f"   Worst performing cluster: {worst}")
            print(f"   Sites in worst cluster: {cluster_analysis.cluster_sizes.get(worst, 0)}")
            print(f"   Average spills in worst cluster: {avg_spills:.2f}")
        
        print("=" * 70 + "\n")
    
    def save_model(self) -> None:
        """Save the clustering model, scaler, and metadata."""
        if self.model is None or self.scaler is None:
            logger.warning("No model or scaler to save")
            return
        
        try:
            # Save model
            joblib.dump(self.model, self.config.model_save_path)
            logger.info(f"Model saved to {self.config.model_save_path}")
            
            # Save scaler
            joblib.dump(self.scaler, self.config.scaler_save_path)
            logger.info(f"Scaler saved to {self.config.scaler_save_path}")
            
            # Save metadata
            metadata = {
                'optimal_k': self.optimal_k,
                'feature_names': self.feature_names,
                'metrics': self.metrics.to_dict() if self.metrics else {},
                'config': self.config.model_dump()
            }
            metadata_path = self.config.model_save_path.replace('.pkl', '_metadata.pkl')
            joblib.dump(metadata, metadata_path)
            logger.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self) -> None:
        """Load a previously saved model."""
        try:
            self.model = joblib.load(self.config.model_save_path)
            self.scaler = joblib.load(self.config.scaler_save_path)
            
            metadata_path = self.config.model_save_path.replace('.pkl', '_metadata.pkl')
            metadata = joblib.load(metadata_path)
            
            self.optimal_k = metadata['optimal_k']
            self.feature_names = metadata['feature_names']
            
            if 'metrics' in metadata:
                self.metrics = ClusterMetrics(**metadata['metrics'])
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_cluster(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Predict clusters for new sites.
        
        Args:
            new_data: DataFrame with new sites
            
        Returns:
            Array of cluster assignments
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        features = new_data[self.feature_names].copy()
        features = features.fillna(features.median())
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        # Ensure return type is ndarray
        if isinstance(predictions, np.ndarray):
            return predictions
        else:
            return np.array(predictions)
    
    def get_high_risk_sites(self) -> Optional[pd.DataFrame]:
        """
        Get sites in the worst-performing cluster.
        
        Returns:
            DataFrame with high-risk sites, or None if not available
        """
        if self.df_clustered is None or self.cluster_analysis is None:
            return None
        
        if self.cluster_analysis.worst_cluster is None:
            return None
        
        # Filter DataFrame - result is always a DataFrame, not a Series
        filtered_df = self.df_clustered[
            self.df_clustered['Cluster'] == self.cluster_analysis.worst_cluster
        ]
        
        # Ensure we return a DataFrame, not a Series
        if isinstance(filtered_df, pd.DataFrame):
            return filtered_df.copy()
        else:
            # This should never happen, but handle edge case
            return pd.DataFrame([filtered_df]) if len(filtered_df) > 0 else pd.DataFrame()


# =============================================================================
# Convenience Function
# =============================================================================

def perform_site_clustering(
    df: pd.DataFrame,
    config: Optional[ClusteringConfig] = None,
    save_model: bool = True,
    create_plots: bool = True,
    print_summary: bool = True
) -> Tuple[Optional[ClusterAnalysisResult], Optional[pd.DataFrame]]:
    """
    Main function to perform site clustering analysis.
    
    This is a convenience function that runs the complete clustering workflow:
    1. Configure and validate parameters
    2. Prepare features
    3. Find optimal number of clusters
    4. Perform K-Means clustering
    5. Evaluate cluster quality
    6. Create visualizations (optional)
    7. Print summary (optional)
    8. Save model (optional)
    
    Args:
        df: Input DataFrame with monitoring sites
        config: ClusteringConfig object (uses defaults if None)
        save_model: Whether to save the model to disk
        create_plots: Whether to create visualizations
        print_summary: Whether to print summary to console
        
    Returns:
        Tuple of (ClusterAnalysisResult, high_risk_sites DataFrame)
        
    Example:
        >>> from scripts.clustering import perform_site_clustering, ClusteringConfig
        >>> 
        >>> config = ClusteringConfig(min_clusters=3, max_clusters=10)
        >>> analysis, high_risk = perform_site_clustering(df, config)
        >>> 
        >>> if analysis:
        ...     print(f"Found {analysis.optimal_k} clusters")
        ...     print(f"Quality: {analysis.metrics.get_quality_interpretation()}")
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting Site Clustering Analysis")
        logger.info("=" * 60)
        
        # Create clusterer
        clusterer = SiteClusterer(config)
        
        # Perform clustering
        success = clusterer.fit(df)
        if not success:
            logger.error("Clustering failed")
            return None, None
        
        # Create visualizations
        if create_plots:
            clusterer.create_visualizations()
        
        # Print summary
        if print_summary:
            clusterer.print_summary()
        
        # Save model
        if save_model:
            clusterer.save_model()
        
        logger.info("Clustering analysis completed successfully!")
        
        return clusterer.cluster_analysis, clusterer.get_high_risk_sites()
        
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
        raise


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    """Example usage demonstration."""
    print("=" * 80)
    print("WATER QUALITY SITE CLUSTERING MODULE")
    print("=" * 80)
    print()
    print("This module provides production-ready clustering functionality.")
    print()
    print("Example usage:")
    print("  from scripts.clustering import SiteClusterer, ClusteringConfig")
    print()
    print("  # Create configuration")
    print("  config = ClusteringConfig(")
    print("      min_clusters=3,")
    print("      max_clusters=10,")
    print("      output_dir='clustering_results'")
    print("  )")
    print()
    print("  # Perform clustering")
    print("  clusterer = SiteClusterer(config)")
    print("  success = clusterer.fit(df_clean)")
    print()
    print("  if success:")
    print("      clusterer.create_visualizations()")
    print("      clusterer.print_summary()")
    print("      clusterer.save_model()")
    print()
    print("=" * 80)
