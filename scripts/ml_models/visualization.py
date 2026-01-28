"""
Visualization Module for ML Models
Purpose: Create visualizations for model evaluation, feature importance, and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from sklearn.model_selection import learning_curve, validation_curve

from .model_utils import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelVisualizer:
    """
    Create visualizations for model evaluation and analysis.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize model visualizer.
        
        Args:
            output_dir: Directory to save plots. If None, uses current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path("plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_learning_curves(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        title: str = "Learning Curves",
        cv: int = 5,
        train_sizes: Optional[List[float]] = None,
        scoring: str = 'r2',
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot learning curves showing training and validation scores.
        
        Args:
            model: Trained model or pipeline
            X: Feature data
            y: Target data
            title: Plot title
            cv: Number of cross-validation folds
            train_sizes: Training set sizes to evaluate
            scoring: Scoring metric
            save_path: Path to save plot. If None, auto-generates.
            
        Returns:
            Path to saved plot
        """
        logger.info("Generating learning curves...")
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            train_sizes=train_sizes,
            scoring=scoring,
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel(f'{scoring.upper()} Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / f"learning_curves_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Learning curves saved to: {save_path}")
        return save_path
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot feature importance scores.
        
        Args:
            feature_importance: Dictionary of feature names to importance scores
            top_n: Number of top features to display
            title: Plot title
            save_path: Path to save plot. If None, auto-generates.
            
        Returns:
            Path to saved plot
        """
        logger.info(f"Plotting top {top_n} feature importances...")
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        if save_path is None:
            save_path = self.output_dir / f"feature_importance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to: {save_path}")
        return save_path
    
    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot. If None, auto-generates.
            
        Returns:
            Path to saved plot
        """
        logger.info("Plotting predictions vs actual...")
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path is None:
            save_path = self.output_dir / f"predictions_vs_actual_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Predictions vs actual plot saved to: {save_path}")
        return save_path
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Analysis",
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot residual analysis (residuals vs predicted, Q-Q plot).
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot. If None, auto-generates.
            
        Returns:
            Path to saved plot
        """
        logger.info("Plotting residual analysis...")
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Check)')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"residuals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Residual analysis plot saved to: {save_path}")
        return save_path
    
    def plot_model_comparison(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        metric: str = 'cv_mean',
        title: str = "Model Comparison",
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot comparison of multiple models.
        
        Args:
            comparison_results: Dictionary of model results from ModelComparator
            metric: Metric to compare ('cv_mean', 'r2', 'rmse', 'mae')
            title: Plot title
            save_path: Path to save plot. If None, auto-generates.
            
        Returns:
            Path to saved plot
        """
        logger.info(f"Plotting model comparison ({metric})...")
        
        models = []
        scores = []
        errors = []
        
        for model_name, results in comparison_results.items():
            if 'error' not in results and metric in results:
                models.append(model_name)
                scores.append(results[metric])
                if f'{metric.replace("_mean", "_std")}' in results:
                    errors.append(results[f'{metric.replace("_mean", "_std")}'])
                else:
                    errors.append(0)
        
        if not models:
            raise ValueError("No valid model results found")
        
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(models))
        
        if any(errors):
            plt.bar(x_pos, scores, yerr=errors, capsize=5, alpha=0.7)
        else:
            plt.bar(x_pos, scores, alpha=0.7)
        
        plt.xlabel('Models')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title)
        plt.xticks(x_pos, models, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to: {save_path}")
        return save_path

