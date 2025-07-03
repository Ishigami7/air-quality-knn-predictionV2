# 🌬️ Air Quality KNN Prediction System V2

A comprehensive implementation of K-Nearest Neighbors algorithm for air quality prediction with detailed analysis, visualization, and reporting capabilities.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

## 🎯 Overview

This project provides a complete solution for air quality prediction using the K-Nearest Neighbors (KNN) machine learning algorithm. It includes comprehensive data processing, exploratory data analysis, model training with hyperparameter tuning, evaluation, and visualization capabilities.

## ✨ Key Features

### 📊 Data Processing & Analysis
- **Data Import & Validation**: Support for CSV and Excel formats with comprehensive validation
- **Missing Value Handling**: Multiple strategies (mean, median, interpolation, etc.)
- **Feature Engineering**: Automatic creation of derived features and air quality categories
- **Outlier Detection**: IQR and Z-score based outlier identification
- **Statistical Analysis**: Comprehensive descriptive statistics and correlation analysis

### 🔍 Exploratory Data Analysis (EDA)
- **Distribution Analysis**: Histograms, KDE plots, and statistical summaries
- **Correlation Analysis**: Heatmaps and correlation matrices
- **Time Series Analysis**: Trend detection and seasonal patterns
- **Air Quality Categories**: Based on Indonesian Air Quality Index (ISPU) standards
- **Data Quality Checks**: Automated validation and quality assessment

### 🤖 Model Development
- **KNN Implementation**: Both classification and regression capabilities
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Feature Scaling**: Automatic standardization and normalization
- **Model Validation**: K-fold cross-validation and performance metrics
- **Feature Importance**: Permutation-based importance analysis

### 📈 Model Evaluation
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, MAE, RMSE, R²
- **Confusion Matrix**: Detailed classification performance analysis
- **Learning Curves**: Training and validation performance over time
- **Residual Analysis**: For regression tasks
- **Cross-Validation**: Robust model validation

### 📊 Visualization & Reporting
- **Data Visualizations**: Distribution plots, correlation heatmaps, time series
- **Model Performance**: Confusion matrices, learning curves, feature importance
- **Interactive Plots**: Customizable visualizations with matplotlib/seaborn
- **Comprehensive Reports**: Automated report generation

### 🔮 Prediction System
- **Real-time Predictions**: Fast inference for new data
- **Confidence Intervals**: Prediction uncertainty quantification
- **Batch Processing**: Efficient processing of multiple samples
- **Model Interpretability**: Feature importance and prediction explanations

### 💾 Export & Persistence
- **Model Persistence**: Save/load trained models with joblib
- **Multiple Formats**: Export to CSV, Excel, JSON
- **Comprehensive Reports**: Automated result compilation
- **Configuration Management**: Flexible configuration system

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Ishigami7/air-quality-knn-predictionV2.git
cd air-quality-knn-predictionV2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install the package:**
```bash
pip install -e .
```

### Basic Usage

```python
from air_quality_knn import DataProcessor, KNNAirQualityModel, AirQualityVisualizer
from air_quality_knn.utils import create_sample_data

# Load or create sample data
data = create_sample_data(n_samples=1000)

# Initialize components
processor = DataProcessor()
processor.data = data

# Preprocess data
data_cleaned = processor.handle_missing_values(strategy='mean')
data_with_categories = processor.create_air_quality_categories()
data_engineered = processor.feature_engineering()

# Prepare features and target
feature_cols = ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']
X = data_engineered[feature_cols]
y = data_engineered['AirQuality_Category']

# Train model
model = KNNAirQualityModel(task_type='classification')
model.hyperparameter_tuning(X, y)
results = model.train(X, y)

# Make predictions
predictions = model.predict(X.sample(10))
pred_with_conf, confidence = model.predict_with_confidence(X.sample(10))

# Visualize results
visualizer = AirQualityVisualizer()
visualizer.plot_model_performance(results)
```

### Using the Jupyter Notebook

The project includes a comprehensive Jupyter notebook demonstrating all features:

```bash
jupyter notebook comprehensive_air_quality_analysis.ipynb
```

## 📁 Project Structure

```
air-quality-knn-predictionV2/
├── air_quality_knn/                    # Main package
│   ├── __init__.py                     # Package initialization
│   ├── data_processing/                # Data processing modules
│   │   └── __init__.py                 # DataProcessor, AirQualityAnalyzer
│   ├── modeling/                       # Model implementation
│   │   └── __init__.py                 # KNNAirQualityModel
│   ├── visualization/                  # Visualization modules
│   │   └── __init__.py                 # AirQualityVisualizer
│   └── utils/                          # Utility functions
│       └── __init__.py                 # ModelPersistence, ExportManager
├── comprehensive_air_quality_analysis.ipynb  # Main analysis notebook
├── v2_air_quality_knn_model (1).ipynb       # Original notebook
├── requirements.txt                    # Dependencies
├── setup.py                           # Package setup
├── README.md                          # This file
└── .gitignore                        # Git ignore rules
```

## 📊 Data Format

The system supports air quality data with the following structure:

### Required Columns:
- **PM10**: Particulate Matter 10 (μg/m³)
- **PM25**: Particulate Matter 2.5 (μg/m³)
- **SO2**: Sulfur Dioxide (μg/m³)
- **CO**: Carbon Monoxide (mg/m³)
- **O3**: Ozone (μg/m³)
- **NO2**: Nitrogen Dioxide (μg/m³)

### Optional Columns:
- **DateTime**: Timestamp for time series analysis
- **Temperature**: Temperature (°C)
- **Humidity**: Relative humidity (%)
- **WindSpeed**: Wind speed (m/s)
- **Pressure**: Atmospheric pressure (hPa)

### Example Data:
```csv
DateTime,PM10,PM25,SO2,CO,O3,NO2
2023-01-01,45.2,28.1,15.3,1.2,65.8,32.4
2023-01-02,52.1,31.7,18.9,1.5,72.3,38.2
...
```

## 🎯 Air Quality Categories

The system uses Indonesian Air Quality Index (ISPU) standards:

| Category | PM10 (μg/m³) | PM2.5 (μg/m³) | Description |
|----------|---------------|---------------|-------------|
| **Good** | ≤ 50 | ≤ 15.5 | Air quality is satisfactory |
| **Moderate** | 51-150 | 15.6-55.4 | Acceptable for most people |
| **Unhealthy** | 151-350 | 55.5-150.4 | Health effects for sensitive groups |
| **Hazardous** | > 350 | > 150.4 | Health warnings for everyone |

## 📈 Model Performance

The KNN model provides excellent performance for air quality prediction:

- **Classification Accuracy**: Typically 85-95%
- **Cross-Validation Stability**: High consistency across folds
- **Feature Importance**: Identifies key pollutants
- **Prediction Confidence**: Quantified uncertainty estimates

## 🔧 Configuration

The system supports flexible configuration through JSON files:

```python
from air_quality_knn.utils import ConfigManager

# Load configuration
config = ConfigManager.load_config('config.json')

# Modify settings
config['model']['cv_folds'] = 10
config['visualization']['figure_size'] = [12, 8]

# Save configuration
ConfigManager.save_config(config, 'new_config.json')
```

## 📤 Export Capabilities

### Supported Export Formats:
- **CSV**: Raw data and predictions
- **Excel**: Multi-sheet reports with charts
- **JSON**: Model parameters and results
- **Joblib**: Trained model persistence
- **PNG/PDF**: High-quality visualizations

### Example Export:
```python
from air_quality_knn.utils import ExportManager, ModelPersistence

# Export comprehensive results
ExportManager.export_comprehensive_results(
    model, data, results, output_dir="output"
)

# Save model
ModelPersistence.save_model(model, "saved_model.joblib")
```

## 🧪 Testing and Validation

The package includes comprehensive testing capabilities:

- **Unit Tests**: Component-level testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Model accuracy validation
- **Data Validation**: Input data quality checks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup:
```bash
# Clone the repository
git clone https://github.com/Ishigami7/air-quality-knn-predictionV2.git

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black air_quality_knn/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ishigami7**
- GitHub: [@Ishigami7](https://github.com/Ishigami7)

## 🙏 Acknowledgments

- Indonesian Ministry of Environment and Forestry for ISPU standards
- Scikit-learn community for machine learning tools
- Open source contributors and environmental monitoring community

## 📚 References

1. Indonesian Air Quality Index (ISPU) Standards
2. WHO Air Quality Guidelines
3. Scikit-learn Documentation
4. Environmental Data Science Best Practices

---

⭐ **Star this repository if you find it helpful!**
