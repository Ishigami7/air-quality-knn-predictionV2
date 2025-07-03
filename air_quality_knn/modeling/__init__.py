"""
KNN Modeling Module for Air Quality Prediction

This module provides comprehensive KNN modeling capabilities including:
- Hyperparameter tuning
- Cross-validation
- Model training and evaluation
- Feature scaling and selection
- Performance metrics and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, learning_curve
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_fscore_support
)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KNNAirQualityModel:
    """
    Comprehensive KNN model for air quality prediction with advanced features
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize KNN model
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.is_fitted = False
        self.best_params = None
        self.cv_scores = None
        self.feature_names = None
        
        # Initialize model based on task type
        if task_type == 'classification':
            self.model = KNeighborsClassifier()
        else:
            self.model = KNeighborsRegressor()
    
    def preprocess_features(self, X: pd.DataFrame, y: pd.Series = None, 
                          fit_transform: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess features with scaling and encoding
        
        Args:
            X: Feature matrix
            y: Target variable
            fit_transform: Whether to fit the preprocessors
            
        Returns:
            Tuple of processed features and target
        """
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        # Handle missing values
        X_processed = X.fillna(X.mean() if hasattr(X, 'mean') else 0)
        
        # Scale features
        if fit_transform:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        # Process target variable
        y_processed = None
        if y is not None:
            if self.task_type == 'classification':
                if fit_transform:
                    y_processed = self.label_encoder.fit_transform(y)
                else:
                    y_processed = self.label_encoder.transform(y)
            else:
                y_processed = y.values if hasattr(y, 'values') else y
        
        return X_scaled, y_processed
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            cv_folds: int = 5, scoring: str = None) -> Dict:
        """
        Perform comprehensive hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for evaluation
            
        Returns:
            Dictionary with tuning results
        """
        logger.info("🔧 Starting hyperparameter tuning...")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_features(X, y, fit_transform=True)
        
        # Define parameter grid
        if self.task_type == 'classification':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
            default_scoring = 'accuracy'
        else:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
            default_scoring = 'neg_mean_squared_error'
        
        scoring = scoring or default_scoring
        
        # Perform grid search
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv_strategy, 
            scoring=scoring, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_processed, y_processed)
        
        # Store results
        self.best_params = grid_search.best_params_
        self.cv_scores = grid_search.cv_results_
        
        # Update model with best parameters
        self.model.set_params(**self.best_params)
        
        results = {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'n_combinations_tested': len(grid_search.cv_results_['params'])
        }
        
        logger.info(f"✅ Hyperparameter tuning completed")
        logger.info(f"🎯 Best parameters: {self.best_params}")
        logger.info(f"📊 Best CV score: {grid_search.best_score_:.4f}")
        
        return results
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train the KNN model
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary with training results
        """
        logger.info("🚀 Starting model training...")
        
        # Split data
        if self.task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Preprocess data
        X_train_processed, y_train_processed = self.preprocess_features(
            X_train, y_train, fit_transform=True
        )
        X_test_processed, y_test_processed = self.preprocess_features(
            X_test, y_test, fit_transform=False
        )
        
        # Train model
        self.model.fit(X_train_processed, y_train_processed)
        self.is_fitted = True
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_processed)
        y_test_pred = self.model.predict(X_test_processed)
        
        # Store data for later use
        self.X_train, self.X_test = X_train_processed, X_test_processed
        self.y_train, self.y_test = y_train_processed, y_test_processed
        self.y_train_pred, self.y_test_pred = y_train_pred, y_test_pred
        
        # Calculate metrics
        results = {
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
        
        if self.task_type == 'classification':
            results.update({
                'train_accuracy': accuracy_score(y_train_processed, y_train_pred),
                'test_accuracy': accuracy_score(y_test_processed, y_test_pred),
                'classification_report': classification_report(
                    y_test_processed, y_test_pred, output_dict=True
                ),
                'confusion_matrix': confusion_matrix(y_test_processed, y_test_pred)
            })
        else:
            results.update({
                'train_mae': mean_absolute_error(y_train_processed, y_train_pred),
                'test_mae': mean_absolute_error(y_test_processed, y_test_pred),
                'train_mse': mean_squared_error(y_train_processed, y_train_pred),
                'test_mse': mean_squared_error(y_test_processed, y_test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train_processed, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test_processed, y_test_pred)),
                'train_r2': r2_score(y_train_processed, y_train_pred),
                'test_r2': r2_score(y_test_processed, y_test_pred)
            })
        
        logger.info("✅ Model training completed")
        return results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5, scoring: str = None) -> Dict:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info("🔄 Performing cross-validation...")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_features(X, y, fit_transform=True)
        
        # Set up scoring
        if scoring is None:
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'] if self.task_type == 'classification' else ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
        
        if isinstance(scoring, str):
            scoring = [scoring]
        
        # Perform cross-validation
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {}
        
        for metric in scoring:
            scores = cross_val_score(
                self.model, X_processed, y_processed, 
                cv=cv_strategy, scoring=metric, n_jobs=-1
            )
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
        
        logger.info("✅ Cross-validation completed")
        return cv_results
    
    def learning_curves(self, X: pd.DataFrame, y: pd.Series, 
                       train_sizes: np.ndarray = None) -> Dict:
        """
        Generate learning curves
        
        Args:
            X: Feature matrix
            y: Target variable
            train_sizes: Array of training sizes to evaluate
            
        Returns:
            Dictionary with learning curve data
        """
        logger.info("📈 Generating learning curves...")
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_features(X, y, fit_transform=True)
        
        # Generate learning curves
        scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        train_sizes_abs, train_scores, val_scores = learning_curve(
            self.model, X_processed, y_processed,
            train_sizes=train_sizes, cv=5, n_jobs=-1,
            scoring=scoring, random_state=42
        )
        
        results = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
        
        logger.info("✅ Learning curves generated")
        return results
    
    def predict(self, X: pd.DataFrame, return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            return_probabilities: Whether to return prediction probabilities (classification only)
            
        Returns:
            Predictions or tuple of predictions and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Please train the model first.")
        
        # Preprocess features
        X_processed, _ = self.preprocess_features(X, fit_transform=False)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Inverse transform predictions if classification
        if self.task_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions)
        
        if return_probabilities and self.task_type == 'classification':
            probabilities = self.model.predict_proba(X_processed)
            return predictions, probabilities
        
        return predictions
    
    def predict_with_confidence(self, X: pd.DataFrame, n_neighbors_confidence: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals
        
        Args:
            X: Feature matrix
            n_neighbors_confidence: Number of neighbors to use for confidence calculation
            
        Returns:
            Tuple of predictions and confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Please train the model first.")
        
        # Preprocess features
        X_processed, _ = self.preprocess_features(X, fit_transform=False)
        
        # Get predictions
        predictions = self.model.predict(X_processed)
        
        # Calculate confidence based on neighbor distances
        if n_neighbors_confidence is None:
            n_neighbors_confidence = min(self.model.n_neighbors, len(self.X_train))
        
        distances, indices = self.model.kneighbors(X_processed, n_neighbors=n_neighbors_confidence)
        
        # Calculate confidence as inverse of mean distance
        confidence_scores = 1 / (1 + distances.mean(axis=1))
        
        # Inverse transform predictions if classification
        if self.task_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions, confidence_scores
    
    def feature_importance(self, X: pd.DataFrame, y: pd.Series, method: str = 'permutation') -> Dict:
        """
        Calculate feature importance
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'permutation' or 'univariate'
            
        Returns:
            Dictionary with feature importance scores
        """
        if method == 'permutation':
            from sklearn.inspection import permutation_importance
            
            # Preprocess data
            X_processed, y_processed = self.preprocess_features(X, y, fit_transform=True)
            
            # Calculate permutation importance
            if not self.is_fitted:
                self.model.fit(X_processed, y_processed)
            
            perm_importance = permutation_importance(
                self.model, X_processed, y_processed, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            
            importance_scores = {
                'feature_names': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std,
                'ranking': np.argsort(perm_importance.importances_mean)[::-1]
            }
            
        elif method == 'univariate':
            # Preprocess data
            X_processed, y_processed = self.preprocess_features(X, y, fit_transform=True)
            
            # Calculate univariate feature scores
            if self.task_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k='all')
            else:
                from sklearn.feature_selection import f_regression
                selector = SelectKBest(score_func=f_regression, k='all')
            
            selector.fit(X_processed, y_processed)
            
            feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            
            importance_scores = {
                'feature_names': feature_names,
                'scores': selector.scores_,
                'p_values': selector.pvalues_,
                'ranking': np.argsort(selector.scores_)[::-1]
            }
        
        logger.info("🎯 Feature importance calculated")
        return importance_scores
    
    def residual_analysis(self) -> Dict:
        """
        Perform residual analysis (for regression tasks)
        
        Returns:
            Dictionary with residual analysis results
        """
        if self.task_type != 'regression':
            logger.warning("⚠️ Residual analysis is only applicable for regression tasks")
            return {}
        
        if not hasattr(self, 'y_test') or not hasattr(self, 'y_test_pred'):
            logger.warning("⚠️ No test predictions available. Please train the model first.")
            return {}
        
        residuals = self.y_test - self.y_test_pred
        
        results = {
            'residuals': residuals,
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'residual_skewness': residuals.std() if len(residuals) > 0 else 0,
            'max_residual': residuals.max() if len(residuals) > 0 else 0,
            'min_residual': residuals.min() if len(residuals) > 0 else 0
        }
        
        # Normality test
        try:
            from scipy.stats import shapiro
            if len(residuals) > 3:
                stat, p_value = shapiro(residuals)
                results['normality_test'] = {'statistic': stat, 'p_value': p_value}
        except ImportError:
            logger.warning("⚠️ Scipy not available for normality test")
        
        logger.info("📊 Residual analysis completed")
        return results
    
    def model_summary(self) -> Dict:
        """
        Generate comprehensive model summary
        
        Returns:
            Dictionary with model summary
        """
        if not self.is_fitted:
            logger.warning("⚠️ Model is not fitted yet")
        
        summary = {
            'task_type': self.task_type,
            'model_parameters': self.model.get_params(),
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'best_hyperparameters': self.best_params
        }
        
        if hasattr(self, 'X_train'):
            summary['training_samples'] = len(self.X_train)
        
        if hasattr(self, 'X_test'):
            summary['test_samples'] = len(self.X_test)
        
        return summary