"""
Air Quality KNN Prediction Package

A comprehensive implementation of K-Nearest Neighbors algorithm 
for air quality prediction with detailed analysis and reporting.

Author: Ishigami7
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Ishigami7"

from .data_processing import DataProcessor, AirQualityAnalyzer
from .modeling import KNNAirQualityModel
from .visualization import AirQualityVisualizer
from .utils import ModelPersistence, ExportManager

__all__ = [
    "DataProcessor",
    "AirQualityAnalyzer", 
    "KNNAirQualityModel",
    "AirQualityVisualizer",
    "ModelPersistence",
    "ExportManager"
]