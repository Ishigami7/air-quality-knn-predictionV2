"""
Data Processing Module for Air Quality KNN Prediction

This module provides comprehensive data processing capabilities including:
- Data import and validation
- Statistical analysis
- Missing value handling
- Feature engineering
- Data quality checks
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Comprehensive data processor for air quality data
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.feature_columns = ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']
        self.meteorological_columns = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure']
        self.required_columns = self.feature_columns
        
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various formats (CSV, Excel)
        
        Args:
            file_path: Path to data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path, **kwargs)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
                
            logger.info(f"✅ Data loaded successfully: {self.data.shape}")
            logger.info(f"📊 Columns: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def validate_data(self) -> Dict:
        """
        Perform comprehensive data validation
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in self.data.columns]
        validation_results['missing_columns'] = missing_columns
        
        # Check data types
        validation_results['data_types'] = self.data.dtypes.to_dict()
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        validation_results['missing_values'] = missing_values[missing_values > 0].to_dict()
        
        # Check for duplicates
        validation_results['duplicate_rows'] = self.data.duplicated().sum()
        
        # Check data range for air quality parameters
        range_issues = {}
        for col in self.feature_columns:
            if col in self.data.columns:
                if (self.data[col] < 0).any():
                    range_issues[col] = "Contains negative values"
                if self.data[col].max() > 1000:  # Reasonable upper bound
                    range_issues[col] = f"Contains very high values (max: {self.data[col].max()})"
        
        validation_results['range_issues'] = range_issues
        
        logger.info("🔍 Data validation completed")
        return validation_results
    
    def handle_missing_values(self, strategy: str = 'mean', custom_values: Dict = None) -> pd.DataFrame:
        """
        Handle missing values using various strategies
        
        Args:
            strategy: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'interpolate', 'custom'
            custom_values: Dictionary of column: value pairs for custom strategy
            
        Returns:
            DataFrame with missing values handled
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        df = self.data.copy()
        
        if strategy == 'mean':
            for col in self.feature_columns:
                if col in df.columns:
                    df[col].fillna(df[col].mean(), inplace=True)
        
        elif strategy == 'median':
            for col in self.feature_columns:
                if col in df.columns:
                    df[col].fillna(df[col].median(), inplace=True)
        
        elif strategy == 'mode':
            for col in self.feature_columns:
                if col in df.columns:
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                    df[col].fillna(mode_value, inplace=True)
        
        elif strategy == 'forward_fill':
            df.fillna(method='ffill', inplace=True)
        
        elif strategy == 'backward_fill':
            df.fillna(method='bfill', inplace=True)
        
        elif strategy == 'interpolate':
            for col in self.feature_columns:
                if col in df.columns:
                    df[col].interpolate(method='linear', inplace=True)
        
        elif strategy == 'custom' and custom_values:
            df.fillna(custom_values, inplace=True)
        
        logger.info(f"✅ Missing values handled using {strategy} strategy")
        return df
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict:
        """
        Detect outliers using various methods
        
        Args:
            method: 'iqr', 'zscore', 'isolation_forest'
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier information
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        outliers_info = {}
        
        for col in self.feature_columns:
            if col not in self.data.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(self.data)) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                outliers = self.data[z_scores > threshold]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(self.data)) * 100
                }
        
        logger.info(f"🔍 Outlier detection completed using {method} method")
        return outliers_info
    
    def create_air_quality_categories(self) -> pd.DataFrame:
        """
        Create air quality categories based on ISPU (Indonesian Air Quality Index) standards
        
        Returns:
            DataFrame with air quality categories
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        df = self.data.copy()
        
        def calculate_air_quality_category(row):
            """Calculate air quality category based on multiple parameters"""
            # ISPU breakpoints for different pollutants
            pm10_val = row.get('PM10', 0)
            pm25_val = row.get('PM25', 0)
            so2_val = row.get('SO2', 0)
            co_val = row.get('CO', 0)
            o3_val = row.get('O3', 0)
            no2_val = row.get('NO2', 0)
            
            # PM10 categories (µg/m³)
            if pm10_val <= 50:
                pm10_cat = 'Good'
            elif pm10_val <= 150:
                pm10_cat = 'Moderate'
            elif pm10_val <= 350:
                pm10_cat = 'Unhealthy'
            else:
                pm10_cat = 'Hazardous'
            
            # PM2.5 categories (µg/m³)
            if pm25_val <= 15.5:
                pm25_cat = 'Good'
            elif pm25_val <= 55.4:
                pm25_cat = 'Moderate'
            elif pm25_val <= 150.4:
                pm25_cat = 'Unhealthy'
            else:
                pm25_cat = 'Hazardous'
            
            # Overall category (worst case)
            categories = [pm10_cat, pm25_cat]
            category_priority = {'Good': 1, 'Moderate': 2, 'Unhealthy': 3, 'Hazardous': 4}
            worst_category = max(categories, key=lambda x: category_priority[x])
            
            return worst_category
        
        df['AirQuality_Category'] = df.apply(calculate_air_quality_category, axis=1)
        
        # Create numerical encoding
        category_mapping = {'Good': 0, 'Moderate': 1, 'Unhealthy': 2, 'Hazardous': 3}
        df['AirQuality_Numeric'] = df['AirQuality_Category'].map(category_mapping)
        
        logger.info("✅ Air quality categories created")
        return df
    
    def feature_engineering(self, include_interactions: bool = True) -> pd.DataFrame:
        """
        Perform feature engineering
        
        Args:
            include_interactions: Whether to include feature interactions
            
        Returns:
            DataFrame with engineered features
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        df = self.data.copy()
        
        # Create ratio features
        if 'PM25' in df.columns and 'PM10' in df.columns:
            df['PM25_PM10_Ratio'] = df['PM25'] / (df['PM10'] + 1e-8)  # Avoid division by zero
        
        # Create combined pollution index
        pollution_cols = [col for col in self.feature_columns if col in df.columns]
        if pollution_cols:
            # Normalize each pollutant to 0-1 scale and take average
            df_normalized = df[pollution_cols].copy()
            for col in pollution_cols:
                df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min() + 1e-8)
            df['Combined_Pollution_Index'] = df_normalized.mean(axis=1)
        
        # Create time-based features if datetime column exists
        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            date_col = date_columns[0]
            df['Hour'] = df[date_col].dt.hour
            df['DayOfWeek'] = df[date_col].dt.dayofweek
            df['Month'] = df[date_col].dt.month
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Create interaction features
        if include_interactions and len(pollution_cols) >= 2:
            df['PM_Interaction'] = df.get('PM10', 0) * df.get('PM25', 0)
            df['Gas_Interaction'] = df.get('SO2', 0) * df.get('NO2', 0)
        
        logger.info("✅ Feature engineering completed")
        return df

class AirQualityAnalyzer:
    """
    Statistical analyzer for air quality data
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.feature_columns = ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']
    
    def descriptive_statistics(self) -> Dict:
        """
        Generate comprehensive descriptive statistics
        
        Returns:
            Dictionary with statistical summaries
        """
        stats = {}
        
        # Basic statistics
        stats['basic'] = self.data.describe()
        
        # Skewness and kurtosis
        from scipy.stats import skew, kurtosis
        stats['skewness'] = {}
        stats['kurtosis'] = {}
        
        for col in self.feature_columns:
            if col in self.data.columns:
                stats['skewness'][col] = skew(self.data[col].dropna())
                stats['kurtosis'][col] = kurtosis(self.data[col].dropna())
        
        # Correlation matrix
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats['correlation'] = self.data[numeric_cols].corr()
        
        logger.info("📊 Descriptive statistics generated")
        return stats
    
    def time_series_analysis(self, date_column: str = None) -> Dict:
        """
        Perform time series analysis if datetime data is available
        
        Args:
            date_column: Name of the datetime column
            
        Returns:
            Dictionary with time series analysis results
        """
        if date_column is None:
            date_columns = self.data.select_dtypes(include=['datetime64']).columns
            if len(date_columns) == 0:
                logger.warning("⚠️ No datetime column found for time series analysis")
                return {}
            date_column = date_columns[0]
        
        results = {}
        
        # Trend analysis
        df_sorted = self.data.sort_values(date_column)
        
        for col in self.feature_columns:
            if col in self.data.columns:
                # Simple trend calculation (correlation with time index)
                time_index = range(len(df_sorted))
                correlation = np.corrcoef(time_index, df_sorted[col].fillna(df_sorted[col].mean()))[0, 1]
                results[f'{col}_trend'] = correlation
        
        # Seasonal patterns (if enough data)
        if len(self.data) > 365:
            # Monthly averages
            monthly_avg = self.data.groupby(self.data[date_column].dt.month)[self.feature_columns].mean()
            results['monthly_patterns'] = monthly_avg
        
        logger.info("📈 Time series analysis completed")
        return results
    
    def correlation_analysis(self) -> Dict:
        """
        Detailed correlation analysis
        
        Returns:
            Dictionary with correlation results
        """
        results = {}
        
        # Feature correlation matrix
        numeric_cols = [col for col in self.feature_columns if col in self.data.columns]
        if len(numeric_cols) > 1:
            correlation_matrix = self.data[numeric_cols].corr()
            results['correlation_matrix'] = correlation_matrix
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Threshold for high correlation
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            results['high_correlation_pairs'] = high_corr_pairs
        
        logger.info("🔗 Correlation analysis completed")
        return results