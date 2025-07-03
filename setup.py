"""
Setup script for Air Quality KNN Prediction Package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="air-quality-knn-prediction",
    version="2.0.0",
    author="Ishigami7",
    author_email="",
    description="Comprehensive K-Nearest Neighbors implementation for air quality prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ishigami7/air-quality-knn-predictionV2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    keywords="air quality, knn, machine learning, prediction, environmental monitoring",
    project_urls={
        "Bug Reports": "https://github.com/Ishigami7/air-quality-knn-predictionV2/issues",
        "Source": "https://github.com/Ishigami7/air-quality-knn-predictionV2",
    },
)