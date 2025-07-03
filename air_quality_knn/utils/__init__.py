"""
Utilities Module for Air Quality KNN Prediction

This module provides utility functions including:
- Model persistence (save/load)
- Data export to various formats
- Logging and configuration
- Helper functions
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPersistence:
    """
    Model persistence utility for saving and loading KNN models
    """
    
    @staticmethod
    def save_model(model, filepath: str, include_metadata: bool = True) -> bool:
        """
        Save KNN model to disk
        
        Args:
            model: Trained KNN model object
            filepath: Path to save the model
            include_metadata: Whether to include metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare model data
            model_data = {
                'model': model.model,
                'scaler': model.scaler,
                'label_encoder': model.label_encoder,
                'is_fitted': model.is_fitted,
                'task_type': model.task_type,
                'feature_names': model.feature_names,
                'best_params': model.best_params,
            }
            
            if include_metadata:
                model_data['metadata'] = {
                    'saved_at': datetime.now().isoformat(),
                    'model_type': 'KNNAirQualityModel',
                    'sklearn_version': joblib.__version__,
                    'feature_count': len(model.feature_names) if model.feature_names else None,
                }
            
            # Save using joblib
            joblib.dump(model_data, filepath)
            
            logger.info(f"✅ Model saved successfully to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    @staticmethod
    def load_model(filepath: str):
        """
        Load KNN model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model object or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Load model data
            model_data = joblib.load(filepath)
            
            # Recreate model object
            from ..modeling import KNNAirQualityModel
            
            # Initialize model
            task_type = model_data.get('task_type', 'classification')
            model = KNNAirQualityModel(task_type=task_type)
            
            # Restore components
            model.model = model_data['model']
            model.scaler = model_data['scaler']
            model.label_encoder = model_data['label_encoder']
            model.is_fitted = model_data.get('is_fitted', False)
            model.feature_names = model_data.get('feature_names')
            model.best_params = model_data.get('best_params')
            
            # Print metadata if available
            if 'metadata' in model_data:
                metadata = model_data['metadata']
                logger.info(f"📋 Model metadata:")
                logger.info(f"   Saved at: {metadata.get('saved_at', 'Unknown')}")
                logger.info(f"   Model type: {metadata.get('model_type', 'Unknown')}")
                logger.info(f"   Features: {metadata.get('feature_count', 'Unknown')}")
            
            logger.info(f"✅ Model loaded successfully from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None
    
    @staticmethod
    def save_model_summary(model, results: Dict, filepath: str) -> bool:
        """
        Save model summary and results
        
        Args:
            model: Trained model object
            results: Training/evaluation results
            filepath: Path to save summary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            summary = {
                'model_info': {
                    'task_type': model.task_type,
                    'is_fitted': model.is_fitted,
                    'feature_count': len(model.feature_names) if model.feature_names else None,
                    'feature_names': model.feature_names,
                    'model_parameters': model.model.get_params() if model.model else None,
                    'best_hyperparameters': model.best_params,
                },
                'performance_metrics': results,
                'timestamp': datetime.now().isoformat(),
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"✅ Model summary saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model summary: {e}")
            return False

class ExportManager:
    """
    Data and results export manager
    """
    
    @staticmethod
    def export_to_csv(data: pd.DataFrame, filepath: str, index: bool = False) -> bool:
        """
        Export DataFrame to CSV
        
        Args:
            data: DataFrame to export
            filepath: Output file path
            index: Whether to include index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            data.to_csv(filepath, index=index)
            logger.info(f"✅ Data exported to CSV: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting to CSV: {e}")
            return False
    
    @staticmethod
    def export_to_excel(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                       filepath: str, index: bool = False) -> bool:
        """
        Export DataFrame(s) to Excel
        
        Args:
            data: DataFrame or dict of DataFrames to export
            filepath: Output file path
            index: Whether to include index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                if isinstance(data, dict):
                    for sheet_name, df in data.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=index)
                else:
                    data.to_excel(writer, sheet_name='Data', index=index)
            
            logger.info(f"✅ Data exported to Excel: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting to Excel: {e}")
            return False
    
    @staticmethod
    def export_results_to_json(results: Dict, filepath: str) -> bool:
        """
        Export results dictionary to JSON
        
        Args:
            results: Results dictionary
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            results_converted = convert_numpy(results)
            
            with open(filepath, 'w') as f:
                json.dump(results_converted, f, indent=2, default=str)
            
            logger.info(f"✅ Results exported to JSON: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting to JSON: {e}")
            return False
    
    @staticmethod
    def create_predictions_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                feature_names: List[str] = None,
                                confidence_scores: np.ndarray = None,
                                filepath: str = None) -> pd.DataFrame:
        """
        Create comprehensive predictions report
        
        Args:
            y_true: True values
            y_pred: Predicted values
            feature_names: Names of features
            confidence_scores: Confidence scores (optional)
            filepath: Path to save report (optional)
            
        Returns:
            DataFrame with prediction report
        """
        try:
            # Create basic report
            report_data = {
                'True_Values': y_true,
                'Predicted_Values': y_pred,
                'Prediction_Error': np.abs(y_true - y_pred) if len(y_true) == len(y_pred) else None
            }
            
            # Add confidence scores if available
            if confidence_scores is not None:
                report_data['Confidence_Score'] = confidence_scores
            
            # Add categorical accuracy for classification
            if hasattr(y_true, 'dtype') and not np.issubdtype(y_true.dtype, np.number):
                report_data['Correct_Prediction'] = (y_true == y_pred)
            
            report_df = pd.DataFrame(report_data)
            
            # Add summary statistics
            if 'Prediction_Error' in report_data and report_data['Prediction_Error'] is not None:
                summary_stats = {
                    'Mean_Absolute_Error': np.mean(report_data['Prediction_Error']),
                    'Max_Error': np.max(report_data['Prediction_Error']),
                    'Min_Error': np.min(report_data['Prediction_Error']),
                    'Std_Error': np.std(report_data['Prediction_Error'])
                }
                
                # Add summary as first row (will be separated in Excel)
                summary_df = pd.DataFrame([summary_stats])
                report_df = pd.concat([summary_df, report_df], ignore_index=True)
            
            # Save if filepath provided
            if filepath:
                ExportManager.export_to_excel(
                    {'Predictions': report_df, 'Summary': summary_df if 'summary_df' in locals() else pd.DataFrame()},
                    filepath
                )
            
            logger.info("✅ Predictions report created")
            return report_df
            
        except Exception as e:
            logger.error(f"❌ Error creating predictions report: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def export_comprehensive_results(model, data: pd.DataFrame, results: Dict,
                                   output_dir: str = "output") -> bool:
        """
        Export comprehensive analysis results
        
        Args:
            model: Trained model object
            data: Original data
            results: Training/evaluation results
            output_dir: Output directory path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export original data
            ExportManager.export_to_csv(
                data, 
                output_dir / f"original_data_{timestamp}.csv"
            )
            
            # Export model summary
            ModelPersistence.save_model_summary(
                model, results, 
                output_dir / f"model_summary_{timestamp}.json"
            )
            
            # Export results as JSON
            ExportManager.export_results_to_json(
                results, 
                output_dir / f"training_results_{timestamp}.json"
            )
            
            # Save model
            ModelPersistence.save_model(
                model, 
                output_dir / f"knn_model_{timestamp}.joblib"
            )
            
            # Export feature importance if available
            try:
                feature_columns = ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']
                available_cols = [col for col in feature_columns if col in data.columns]
                
                if available_cols:
                    importance_data = model.feature_importance(data[available_cols], data.get('AirQuality_Numeric', data.iloc[:, -1]))
                    ExportManager.export_results_to_json(
                        importance_data,
                        output_dir / f"feature_importance_{timestamp}.json"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Could not export feature importance: {e}")
            
            # Create README with export information
            readme_content = f"""# Air Quality KNN Model Export
            
Export created on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Files Included:

1. **original_data_{timestamp}.csv** - Original dataset used for training
2. **model_summary_{timestamp}.json** - Comprehensive model information and parameters
3. **training_results_{timestamp}.json** - Training and evaluation metrics
4. **knn_model_{timestamp}.joblib** - Trained model (can be loaded using joblib)
5. **feature_importance_{timestamp}.json** - Feature importance analysis (if available)

## Model Information:
- Task Type: {model.task_type}
- Model Status: {'Fitted' if model.is_fitted else 'Not Fitted'}
- Feature Count: {len(model.feature_names) if model.feature_names else 'Unknown'}
- Best Parameters: {model.best_params}

## Usage:
To load the model:
```python
import joblib
from air_quality_knn.utils import ModelPersistence

model = ModelPersistence.load_model('knn_model_{timestamp}.joblib')
```

## Performance Summary:
{json.dumps(results, indent=2, default=str)}
"""
            
            with open(output_dir / f"README_{timestamp}.md", 'w') as f:
                f.write(readme_content)
            
            logger.info(f"✅ Comprehensive results exported to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting comprehensive results: {e}")
            return False

class ConfigManager:
    """
    Configuration management utility
    """
    
    DEFAULT_CONFIG = {
        'model': {
            'task_type': 'classification',
            'hyperparameter_tuning': True,
            'cv_folds': 5,
            'test_size': 0.2,
            'random_state': 42
        },
        'data': {
            'feature_columns': ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2'],
            'target_column': 'AirQuality_Category',
            'missing_value_strategy': 'mean'
        },
        'visualization': {
            'figure_size': [15, 10],
            'save_plots': True,
            'plot_format': 'png',
            'dpi': 300
        },
        'export': {
            'save_model': True,
            'export_predictions': True,
            'export_summary': True,
            'output_directory': 'output'
        }
    }
    
    @staticmethod
    def load_config(filepath: str = None) -> Dict:
        """
        Load configuration from file or return default
        
        Args:
            filepath: Path to config file
            
        Returns:
            Configuration dictionary
        """
        if filepath and Path(filepath).exists():
            try:
                with open(filepath, 'r') as f:
                    config = json.load(f)
                logger.info(f"✅ Configuration loaded from {filepath}")
                return config
            except Exception as e:
                logger.warning(f"⚠️ Error loading config: {e}. Using default.")
        
        return ConfigManager.DEFAULT_CONFIG.copy()
    
    @staticmethod
    def save_config(config: Dict, filepath: str) -> bool:
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ Configuration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving config: {e}")
            return False

class Logger:
    """
    Enhanced logging utility
    """
    
    @staticmethod
    def setup_logging(log_file: str = None, level: str = 'INFO') -> logging.Logger:
        """
        Setup enhanced logging
        
        Args:
            log_file: Path to log file (optional)
            level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger('air_quality_knn')
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_system_info():
        """Log system information"""
        import platform
        import sys
        
        logger.info("🖥️ System Information:")
        logger.info(f"   Platform: {platform.platform()}")
        logger.info(f"   Python: {sys.version}")
        logger.info(f"   Architecture: {platform.architecture()}")

def create_sample_data(n_samples: int = 1000, noise_level: float = 0.1) -> pd.DataFrame:
    """
    Create sample air quality data for testing
    
    Args:
        n_samples: Number of samples to generate
        noise_level: Level of noise to add
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    # Generate base patterns
    time_index = np.arange(n_samples)
    
    # Create realistic air quality patterns
    pm10 = 50 + 30 * np.sin(time_index * 2 * np.pi / 365) + np.random.normal(0, 20, n_samples)
    pm25 = pm10 * 0.6 + np.random.normal(0, 10, n_samples)
    so2 = 20 + 10 * np.sin(time_index * 2 * np.pi / 365 + np.pi/4) + np.random.normal(0, 5, n_samples)
    co = 1.5 + 0.5 * np.sin(time_index * 2 * np.pi / 365 + np.pi/2) + np.random.normal(0, 0.3, n_samples)
    o3 = 60 + 20 * np.sin(time_index * 2 * np.pi / 365 + np.pi) + np.random.normal(0, 15, n_samples)
    no2 = 30 + 15 * np.sin(time_index * 2 * np.pi / 365 + np.pi/3) + np.random.normal(0, 8, n_samples)
    
    # Ensure positive values
    pm10 = np.maximum(pm10, 0)
    pm25 = np.maximum(pm25, 0)
    so2 = np.maximum(so2, 0)
    co = np.maximum(co, 0)
    o3 = np.maximum(o3, 0)
    no2 = np.maximum(no2, 0)
    
    # Create DataFrame
    data = pd.DataFrame({
        'DateTime': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'PM10': pm10,
        'PM25': pm25,
        'SO2': so2,
        'CO': co,
        'O3': o3,
        'NO2': no2
    })
    
    # Add some missing values randomly
    missing_rate = 0.02  # 2% missing values
    for col in ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']:
        missing_indices = np.random.choice(n_samples, int(n_samples * missing_rate), replace=False)
        data.loc[missing_indices, col] = np.nan
    
    logger.info(f"✅ Sample data created with {n_samples} samples")
    return data