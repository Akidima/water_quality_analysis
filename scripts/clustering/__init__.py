"""
Water Quality Site Clustering Module
====================================

This module provides production-ready clustering utilities for water quality
monitoring sites. It includes Pydantic-validated configuration models,
K-Means clustering with automatic optimal k detection, and comprehensive
visualization capabilities.

Main Components
---------------
- ClusteringConfig: Pydantic-validated configuration for clustering parameters
- ClusterMetrics: Validated model for cluster quality metrics
- ClusterAnalysisResult: Comprehensive result model for clustering analysis
- ClusterValidator: Utility class for validating clustering data and results
- OptimalClusterFinder: Automatic detection of optimal cluster count
- SiteClusterer: Main class for performing site clustering
- perform_site_clustering: Convenience function for complete workflow

Quick Start
-----------
Basic usage with default configuration:

    >>> from scripts.clustering import SiteClusterer
    >>> 
    >>> clusterer = SiteClusterer()
    >>> success = clusterer.fit(df)
    >>> if success:
    ...     clusterer.create_visualizations()
    ...     clusterer.print_summary()

Custom configuration:

    >>> from scripts.clustering import SiteClusterer, ClusteringConfig
    >>> 
    >>> config = ClusteringConfig(
    ...     min_clusters=3,
    ...     max_clusters=10,
    ...     output_dir='my_results'
    ... )
    >>> clusterer = SiteClusterer(config)
    >>> success = clusterer.fit(df)

Using the convenience function:

    >>> from scripts.clustering import perform_site_clustering, ClusteringConfig
    >>> 
    >>> config = ClusteringConfig(min_clusters=3, max_clusters=8)
    >>> analysis, high_risk_sites = perform_site_clustering(df, config)
    >>> 
    >>> if analysis:
    ...     print(f"Found {analysis.optimal_k} clusters")
    ...     print(f"Worst cluster: {analysis.worst_cluster}")

Configuration Options
--------------------
ClusteringConfig supports the following parameters:

- output_dir (str): Directory for saving plots (default: 'plots')
- model_save_path (str): Path for model file (default: 'models/clustering_model.pkl')
- scaler_save_path (str): Path for scaler file (default: 'models/clustering_scaler.pkl')
- min_clusters (int): Minimum clusters to try (default: 2, must be >= 2)
- max_clusters (int): Maximum clusters to try (default: 8, must be <= 50)
- random_state (int): Random seed for reproducibility (default: 42)
- n_init (int): Number of K-Means initializations (default: 10)
- max_iter (int): Maximum iterations per run (default: 300)
- base_features (List[str]): Required feature columns
- optional_features (List[str]): Optional feature columns
- figure_size (Tuple[int, int]): Figure size for plots (default: (16, 12))
- dpi (int): DPI for saved figures (default: 300)

Cluster Quality Metrics
-----------------------
The module calculates and validates the following metrics:

- Silhouette Score: Measures cluster separation (-1 to 1, higher is better)
- Davies-Bouldin Score: Measures cluster similarity (0 to inf, lower is better)
- Calinski-Harabasz Score: Ratio of between/within variance (higher is better)

Quality interpretations:
- Silhouette > 0.7: Excellent clusters
- Silhouette > 0.5: Good clusters
- Silhouette > 0.3: Fair clusters
- Silhouette <= 0.3: Poor clusters

Dependencies
------------
Required:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- pydantic
- joblib

Optional:
- kneed (for automatic elbow detection)

Notes
-----
This module uses Pydantic for configuration validation, ensuring type safety
and parameter validation at runtime. Invalid configurations will raise
ValidationError with detailed error messages.

The clustering uses the K-Means algorithm with the elbow method for optimal
k detection. If the kneed library is not available, a fallback method using
the middle of the cluster range is used.

See Also
--------
- scripts.pydantic_enhancements: Additional Pydantic validators
- scripts.ml_models: Machine learning model utilities

Version History
---------------
v1.0.0 (2026-02): Initial release with Pydantic enhancements
"""

__version__ = "1.0.0"
__author__ = "Water Quality Analysis Team"

# Import main classes and functions from cluster_utils
from .cluster_utils import (
    # Configuration Models (Pydantic)
    ClusteringConfig,
    ClusterMetrics,
    ClusterAnalysisResult,
    
    # Validation Utilities
    ClusterValidator,
    
    # Core Clustering Classes
    OptimalClusterFinder,
    SiteClusterer,
    
    # Convenience Function
    perform_site_clustering,
)

# Define public API
__all__ = [
    # Configuration Models
    "ClusteringConfig",
    "ClusterMetrics",
    "ClusterAnalysisResult",
    
    # Validation
    "ClusterValidator",
    
    # Core Classes
    "OptimalClusterFinder",
    "SiteClusterer",
    
    # Functions
    "perform_site_clustering",
    
    # Module metadata
    "__version__",
    "__author__",
]


def get_version() -> str:
    """Return the module version string."""
    return __version__


def get_available_features() -> dict:
    """
    Get information about available features and dependencies.
    
    Returns:
        Dictionary with feature availability information
    """
    from .cluster_utils import KNEED_AVAILABLE
    
    return {
        "version": __version__,
        "kneed_available": KNEED_AVAILABLE,
        "elbow_detection": "automatic" if KNEED_AVAILABLE else "fallback",
        "pydantic_validation": True,
        "supported_metrics": [
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score"
        ],
        "clustering_algorithm": "KMeans"
    }
