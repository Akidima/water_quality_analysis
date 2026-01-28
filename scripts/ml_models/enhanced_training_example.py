"""
Enhanced Training Example
Demonstrates all new enhancements: hyperparameter tuning, model comparison, feature engineering, and visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import cast

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_models.model_utils import ModelConfig, get_logger
from scripts.ml_models.hyperparameter_tuning import HyperparameterTuner, BayesianOptimizer
from scripts.ml_models.model_comparison import ModelComparator
from scripts.ml_models.advanced_feature_engineering import AdvancedFeatureEngineer
from scripts.ml_models.visualization import ModelVisualizer
from scripts.ml_models.train_pipeline import SpillTrainingPipeline

logger = get_logger(__name__)


def main():
    """Demonstrate all enhancements."""
    
    print("=" * 80)
    print("ENHANCED MODELING PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Load data
    data_file = PROJECT_ROOT / "export" / "cleaned_data" / "cleaned_water_data.csv"
    if not data_file.exists():
        data_file = PROJECT_ROOT / "data" / "national_water_plan.csv"
    
    print(f"\n1. Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Shape: {df.shape}")
    
    # Configure model
    config = ModelConfig(
        features=['Spill Events 2020', 'Spill Events 2021', 'Spill Events 2022', 'Latitude', 'Longitude'],
        target='Predicted Annual Spill Frequence Post Scheme',
        n_estimators=100,
        random_state=42,
        test_size=0.2
    )
    
    # Prepare data
    print("\n2. Preparing data...")
    required_cols = config.features + [config.target]
    clean_df = df[required_cols].dropna()
    print(f"   Clean data shape: {clean_df.shape}")
    
    X = cast(pd.DataFrame, clean_df[config.features])
    y = cast(pd.Series, clean_df[config.target])
    
    # Advanced Feature Engineering
    print("\n3. Advanced Feature Engineering...")
    feature_engineer = AdvancedFeatureEngineer()
    
    # Create interaction features
    feature_df = cast(pd.DataFrame, clean_df[config.features])
    df_with_interactions = feature_engineer.create_interaction_features(
        feature_df,
        columns=config.features
    )
    print(f"   Created {len(df_with_interactions.columns) - len(config.features)} interaction features")
    
    # Feature selection
    X_selected, selected_features, importance = feature_engineer.select_features_importance(
        X, y, max_features=7
    )
    print(f"   Selected {len(selected_features)} features based on importance")
    
    # Model Comparison
    print("\n4. Comparing Multiple Models...")
    comparator = ModelComparator(config)
    comparison_results = comparator.compare_models(
        X_selected, y,
        model_names=['random_forest', 'gradient_boosting', 'linear_regression', 'ridge'],
        cv=5
    )
    
    best_model_name, best_model = comparator.get_best_model()
    print(f"   Best model: {best_model_name}")
    
    # Hyperparameter Tuning
    print("\n5. Hyperparameter Tuning...")
    tuner = HyperparameterTuner(config, model_type='random_forest')
    
    # Grid search
    grid_results = tuner.grid_search(
        X_selected, y,
        cv=3,  # Reduced for demo
        scoring='r2'
    )
    print(f"   Best parameters: {grid_results['best_params']}")
    print(f"   Best score: {grid_results['best_score']:.4f}")
    
    # Visualization
    print("\n6. Creating Visualizations...")
    visualizer = ModelVisualizer(output_dir=PROJECT_ROOT / "plots")
    
    # Train final model
    trainer = SpillTrainingPipeline(config=config)
    trainer.pipeline = tuner.get_best_estimator()
    trainer._set_trained(True)
    
    # Predictions for visualization
    y_pred = trainer.predict(X_selected)
    
    # Create plots
    plot_paths = []
    
    # Feature importance
    if importance:
        path = visualizer.plot_feature_importance(importance, top_n=10)
        plot_paths.append(path)
        print(f"   ✓ Feature importance plot: {path}")
    
    # Predictions vs actual
    y_true_arr = np.asarray(y.values)
    y_pred_arr = np.asarray(y_pred)
    path = visualizer.plot_prediction_vs_actual(y_true_arr, y_pred_arr)
    plot_paths.append(path)
    print(f"   ✓ Predictions vs actual plot: {path}")
    
    # Residuals
    path = visualizer.plot_residuals(y_true_arr, y_pred_arr)
    plot_paths.append(path)
    print(f"   ✓ Residual analysis plot: {path}")
    
    # Model comparison
    path = visualizer.plot_model_comparison(comparison_results, metric='cv_mean')
    plot_paths.append(path)
    print(f"   ✓ Model comparison plot: {path}")
    
    print("\n" + "=" * 80)
    print("✅ ENHANCEMENT DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated {len(plot_paths)} visualization files")
    print(f"Best model: {best_model_name}")
    print(f"Best CV score: {grid_results['best_score']:.4f}")
    
    return {
        'best_model': best_model_name,
        'best_score': grid_results['best_score'],
        'selected_features': selected_features,
        'plot_paths': plot_paths
    }


if __name__ == "__main__":
    results = main()

