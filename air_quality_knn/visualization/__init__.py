"""
Visualization Module for Air Quality KNN Prediction

This module provides comprehensive visualization capabilities including:
- Data distribution plots
- Correlation visualizations
- Model performance plots
- Interactive visualizations
- Time series plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirQualityVisualizer:
    """
    Comprehensive visualization class for air quality data and model results
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.setup_style()
        
    def setup_style(self):
        """Setup matplotlib and seaborn styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_data_overview(self, data: pd.DataFrame, feature_columns: List[str] = None) -> plt.Figure:
        """
        Create comprehensive data overview plots
        
        Args:
            data: DataFrame to visualize
            feature_columns: List of feature columns to plot
            
        Returns:
            Matplotlib figure
        """
        if feature_columns is None:
            feature_columns = ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']
        
        # Filter available columns
        available_cols = [col for col in feature_columns if col in data.columns]
        
        if not available_cols:
            logger.warning("⚠️ No specified feature columns found in data")
            return None
        
        n_cols = min(3, len(available_cols))
        n_rows = (len(available_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle('📊 Air Quality Parameters Distribution', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(available_cols):
            ax = axes[i]
            
            # Histogram with KDE
            data[col].dropna().hist(bins=30, alpha=0.7, ax=ax, density=True)
            data[col].dropna().plot.kde(ax=ax, color='red', linewidth=2)
            
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(f'{col} (μg/m³)')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Mean: {data[col].mean():.2f}\nStd: {data[col].std():.2f}\nSkew: {data[col].skew():.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Remove extra subplots
        for i in range(len(available_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        logger.info("📊 Data overview plots created")
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame, feature_columns: List[str] = None) -> plt.Figure:
        """
        Create correlation matrix heatmap
        
        Args:
            data: DataFrame to analyze
            feature_columns: List of feature columns
            
        Returns:
            Matplotlib figure
        """
        if feature_columns is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in feature_columns if col in data.columns]
        
        if len(numeric_cols) < 2:
            logger.warning("⚠️ Not enough numeric columns for correlation matrix")
            return None
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('🔗 Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        logger.info("🔗 Correlation matrix created")
        return fig
    
    def plot_air_quality_categories(self, data: pd.DataFrame, category_column: str = 'AirQuality_Category') -> plt.Figure:
        """
        Plot air quality category distribution
        
        Args:
            data: DataFrame with air quality categories
            category_column: Name of category column
            
        Returns:
            Matplotlib figure
        """
        if category_column not in data.columns:
            logger.warning(f"⚠️ Column {category_column} not found in data")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('🎯 Air Quality Category Analysis', fontsize=16, fontweight='bold')
        
        # Count plot
        category_counts = data[category_column].value_counts()
        colors = ['green', 'yellow', 'orange', 'red'][:len(category_counts)]
        
        ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Distribution of Air Quality Categories')
        
        # Bar plot
        sns.countplot(data=data, x=category_column, ax=ax2, palette=colors)
        ax2.set_title('Count of Air Quality Categories')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add counts on bars
        for i, v in enumerate(category_counts.values):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        logger.info("🎯 Air quality category plots created")
        return fig
    
    def plot_time_series(self, data: pd.DataFrame, date_column: str, 
                        feature_columns: List[str] = None) -> plt.Figure:
        """
        Create time series plots
        
        Args:
            data: DataFrame with time series data
            date_column: Name of datetime column
            feature_columns: List of features to plot
            
        Returns:
            Matplotlib figure
        """
        if date_column not in data.columns:
            logger.warning(f"⚠️ Date column {date_column} not found in data")
            return None
        
        if feature_columns is None:
            feature_columns = ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']
        
        available_cols = [col for col in feature_columns if col in data.columns]
        
        if not available_cols:
            logger.warning("⚠️ No specified feature columns found in data")
            return None
        
        n_plots = len(available_cols)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4*n_plots))
        fig.suptitle('📈 Time Series Analysis of Air Quality Parameters', fontsize=16, fontweight='bold')
        
        if n_plots == 1:
            axes = [axes]
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Sort by date
        data_sorted = data.sort_values(date_column)
        
        for i, col in enumerate(available_cols):
            ax = axes[i]
            
            # Plot time series
            ax.plot(data_sorted[date_column], data_sorted[col], linewidth=1, alpha=0.7)
            
            # Add rolling average
            if len(data_sorted) > 7:
                rolling_avg = data_sorted[col].rolling(window=7, center=True).mean()
                ax.plot(data_sorted[date_column], rolling_avg, color='red', linewidth=2, 
                       label='7-day Moving Average')
            
            ax.set_title(f'{col} Over Time')
            ax.set_ylabel(f'{col} (μg/m³)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if i == len(available_cols) - 1:
                ax.set_xlabel('Date')
        
        plt.tight_layout()
        logger.info("📈 Time series plots created")
        return fig
    
    def plot_model_performance(self, model_results: Dict, task_type: str = 'classification') -> plt.Figure:
        """
        Plot model performance metrics
        
        Args:
            model_results: Dictionary with model training results
            task_type: 'classification' or 'regression'
            
        Returns:
            Matplotlib figure
        """
        if task_type == 'classification':
            return self._plot_classification_performance(model_results)
        else:
            return self._plot_regression_performance(model_results)
    
    def _plot_classification_performance(self, results: Dict) -> plt.Figure:
        """Plot classification performance metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 Classification Model Performance', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        if 'train_accuracy' in results and 'test_accuracy' in results:
            accuracies = [results['train_accuracy'], results['test_accuracy']]
            ax1.bar(['Training', 'Testing'], accuracies, color=['blue', 'orange'])
            ax1.set_title('Training vs Testing Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim([0, 1])
            
            # Add value labels on bars
            for i, v in enumerate(accuracies):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Confusion Matrix
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title('Confusion Matrix')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
        
        # Classification Report (F1-scores)
        if 'classification_report' in results:
            report = results['classification_report']
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            f1_scores = [report[cls]['f1-score'] for cls in classes]
            
            ax3.bar(classes, f1_scores, color='green')
            ax3.set_title('F1-Score by Class')
            ax3.set_ylabel('F1-Score')
            ax3.set_xlabel('Classes')
            ax3.tick_params(axis='x', rotation=45)
        
        # Precision, Recall, F1-Score comparison
        if 'classification_report' in results:
            report = results['classification_report']
            metrics = ['precision', 'recall', 'f1-score']
            macro_values = [report['macro avg'][metric] for metric in metrics]
            
            ax4.bar(metrics, macro_values, color=['red', 'blue', 'green'])
            ax4.set_title('Macro Average Metrics')
            ax4.set_ylabel('Score')
            ax4.set_ylim([0, 1])
            
            # Add value labels on bars
            for i, v in enumerate(macro_values):
                ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def _plot_regression_performance(self, results: Dict) -> plt.Figure:
        """Plot regression performance metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Regression Model Performance', fontsize=16, fontweight='bold')
        
        # MAE comparison
        if 'train_mae' in results and 'test_mae' in results:
            mae_values = [results['train_mae'], results['test_mae']]
            ax1.bar(['Training', 'Testing'], mae_values, color=['blue', 'orange'])
            ax1.set_title('Mean Absolute Error')
            ax1.set_ylabel('MAE')
            
            for i, v in enumerate(mae_values):
                ax1.text(i, v + max(mae_values)*0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # RMSE comparison
        if 'train_rmse' in results and 'test_rmse' in results:
            rmse_values = [results['train_rmse'], results['test_rmse']]
            ax2.bar(['Training', 'Testing'], rmse_values, color=['green', 'red'])
            ax2.set_title('Root Mean Square Error')
            ax2.set_ylabel('RMSE')
            
            for i, v in enumerate(rmse_values):
                ax2.text(i, v + max(rmse_values)*0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # R² comparison
        if 'train_r2' in results and 'test_r2' in results:
            r2_values = [results['train_r2'], results['test_r2']]
            ax3.bar(['Training', 'Testing'], r2_values, color=['purple', 'brown'])
            ax3.set_title('R² Score')
            ax3.set_ylabel('R²')
            ax3.set_ylim([0, 1])
            
            for i, v in enumerate(r2_values):
                ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # All metrics comparison
        metrics = []
        train_values = []
        test_values = []
        
        if 'train_mae' in results:
            metrics.append('MAE')
            train_values.append(results['train_mae'])
            test_values.append(results['test_mae'])
        if 'train_mse' in results:
            metrics.append('MSE')
            train_values.append(results['train_mse'])
            test_values.append(results['test_mse'])
        if 'train_r2' in results:
            metrics.append('R²')
            train_values.append(results['train_r2'])
            test_values.append(results['test_r2'])
        
        if metrics:
            x = np.arange(len(metrics))
            width = 0.35
            
            ax4.bar(x - width/2, train_values, width, label='Training', color='blue', alpha=0.7)
            ax4.bar(x + width/2, test_values, width, label='Testing', color='orange', alpha=0.7)
            
            ax4.set_title('All Metrics Comparison')
            ax4.set_ylabel('Score')
            ax4.set_xlabel('Metrics')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, learning_curve_data: Dict) -> plt.Figure:
        """
        Plot learning curves
        
        Args:
            learning_curve_data: Dictionary with learning curve data
            
        Returns:
            Matplotlib figure
        """
        if not learning_curve_data:
            logger.warning("⚠️ No learning curve data provided")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_sizes = learning_curve_data['train_sizes']
        train_scores_mean = learning_curve_data['train_scores_mean']
        train_scores_std = learning_curve_data['train_scores_std']
        val_scores_mean = learning_curve_data['val_scores_mean']
        val_scores_std = learning_curve_data['val_scores_std']
        
        # Plot learning curves
        ax.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                       train_scores_mean + train_scores_std, alpha=0.1, color='blue')
        
        ax.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross-validation score')
        ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                       val_scores_mean + val_scores_std, alpha=0.1, color='red')
        
        ax.set_title('📈 Learning Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        logger.info("📈 Learning curves plotted")
        return fig
    
    def plot_feature_importance(self, importance_data: Dict, top_k: int = 10) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            importance_data: Dictionary with feature importance data
            top_k: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if not importance_data:
            logger.warning("⚠️ No feature importance data provided")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        feature_names = importance_data['feature_names']
        
        if 'importance_mean' in importance_data:
            importance_scores = importance_data['importance_mean']
            importance_std = importance_data.get('importance_std', None)
        else:
            importance_scores = importance_data['scores']
            importance_std = None
        
        # Get top k features
        top_indices = np.argsort(importance_scores)[-top_k:]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]
        top_std = importance_std[top_indices] if importance_std is not None else None
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_scores, xerr=top_std, align='center', alpha=0.7)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'🎯 Top {top_k} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        logger.info("🎯 Feature importance plotted")
        return fig
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  task_type: str = 'classification') -> plt.Figure:
        """
        Plot predictions vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: 'classification' or 'regression'
            
        Returns:
            Matplotlib figure
        """
        if task_type == 'regression':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('🎯 Predictions vs Actual Values', fontsize=16, fontweight='bold')
            
            # Scatter plot
            ax1.scatter(y_true, y_pred, alpha=0.6)
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Predicted vs Actual')
            ax1.grid(True, alpha=0.3)
            
            # Residuals plot
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
            ax2.grid(True, alpha=0.3)
            
        else:
            # For classification, show confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('🎯 Confusion Matrix', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        logger.info("🎯 Predictions vs actual plotted")
        return fig
    
    def plot_hyperparameter_tuning(self, cv_results: Dict, param_name: str = 'n_neighbors') -> plt.Figure:
        """
        Plot hyperparameter tuning results
        
        Args:
            cv_results: Cross-validation results from GridSearchCV
            param_name: Parameter name to plot
            
        Returns:
            Matplotlib figure
        """
        if not cv_results or 'params' not in cv_results:
            logger.warning("⚠️ No hyperparameter tuning results provided")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract parameter values and scores
        param_values = [params[param_name] for params in cv_results['params'] if param_name in params]
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']
        
        if not param_values:
            logger.warning(f"⚠️ Parameter {param_name} not found in results")
            return None
        
        # Group by parameter value
        unique_values = sorted(list(set(param_values)))
        grouped_scores = []
        grouped_stds = []
        
        for value in unique_values:
            indices = [i for i, v in enumerate(param_values) if v == value]
            scores = [mean_scores[i] for i in indices]
            stds = [std_scores[i] for i in indices]
            grouped_scores.append(np.mean(scores))
            grouped_stds.append(np.mean(stds))
        
        # Plot
        ax.errorbar(unique_values, grouped_scores, yerr=grouped_stds, 
                   fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Cross-Validation Score')
        ax.set_title(f'🔧 Hyperparameter Tuning: {param_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Highlight best value
        best_idx = np.argmax(grouped_scores)
        ax.scatter(unique_values[best_idx], grouped_scores[best_idx], 
                  color='red', s=100, zorder=5, label=f'Best: {unique_values[best_idx]}')
        ax.legend()
        
        plt.tight_layout()
        logger.info("🔧 Hyperparameter tuning results plotted")
        return fig
    
    def create_comprehensive_report(self, data: pd.DataFrame, model_results: Dict,
                                  feature_columns: List[str] = None, 
                                  save_path: str = None) -> List[plt.Figure]:
        """
        Create comprehensive visualization report
        
        Args:
            data: Input data
            model_results: Model training results
            feature_columns: List of feature columns
            save_path: Path to save figures (optional)
            
        Returns:
            List of matplotlib figures
        """
        figures = []
        
        # Data overview
        fig1 = self.plot_data_overview(data, feature_columns)
        if fig1:
            figures.append(fig1)
            if save_path:
                fig1.savefig(f"{save_path}_data_overview.png", dpi=300, bbox_inches='tight')
        
        # Correlation matrix
        fig2 = self.plot_correlation_matrix(data, feature_columns)
        if fig2:
            figures.append(fig2)
            if save_path:
                fig2.savefig(f"{save_path}_correlation.png", dpi=300, bbox_inches='tight')
        
        # Air quality categories (if available)
        if 'AirQuality_Category' in data.columns:
            fig3 = self.plot_air_quality_categories(data)
            if fig3:
                figures.append(fig3)
                if save_path:
                    fig3.savefig(f"{save_path}_categories.png", dpi=300, bbox_inches='tight')
        
        # Model performance
        task_type = 'classification' if 'train_accuracy' in model_results else 'regression'
        fig4 = self.plot_model_performance(model_results, task_type)
        if fig4:
            figures.append(fig4)
            if save_path:
                fig4.savefig(f"{save_path}_performance.png", dpi=300, bbox_inches='tight')
        
        logger.info(f"📊 Comprehensive report created with {len(figures)} figures")
        return figures