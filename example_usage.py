#!/usr/bin/env python3
"""
Example Script: Air Quality KNN Prediction
==========================================

This script demonstrates the basic usage of the Air Quality KNN Prediction system.

Author: Ishigami7
Version: 2.0.0
"""

import warnings
warnings.filterwarnings('ignore')

from air_quality_knn.data_processing import DataProcessor, AirQualityAnalyzer
from air_quality_knn.modeling import KNNAirQualityModel
from air_quality_knn.visualization import AirQualityVisualizer
from air_quality_knn.utils import ModelPersistence, ExportManager, create_sample_data

def main():
    print("🌬️ Air Quality KNN Prediction Example")
    print("=" * 50)
    
    # Step 1: Create or Load Data
    print("\n📊 Step 1: Loading Data")
    print("-" * 30)
    
    # For this example, we'll create sample data
    # In real usage, you would load your own data:
    # processor = DataProcessor()
    # data = processor.load_data('your_data.csv')
    
    data = create_sample_data(n_samples=500)
    print(f"✅ Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"📋 Columns: {list(data.columns)}")
    
    # Step 2: Data Processing
    print("\n🧹 Step 2: Data Processing")
    print("-" * 30)
    
    processor = DataProcessor()
    processor.data = data
    
    # Validate data
    validation_results = processor.validate_data()
    print(f"📊 Missing values: {len(validation_results['missing_values'])} columns affected")
    
    # Handle missing values
    data_cleaned = processor.handle_missing_values(strategy='mean')
    
    # Create air quality categories
    data_with_categories = processor.create_air_quality_categories()
    processor.data = data_with_categories
    
    # Feature engineering
    data_engineered = processor.feature_engineering(include_interactions=True)
    print(f"✅ Data processed: {data_engineered.shape[1]} total features")
    
    # Step 3: Exploratory Analysis
    print("\n📈 Step 3: Exploratory Analysis")
    print("-" * 30)
    
    analyzer = AirQualityAnalyzer(data_engineered)
    stats = analyzer.descriptive_statistics()
    
    print("📊 Air Quality Category Distribution:")
    category_counts = data_engineered['AirQuality_Category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(data_engineered)) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    # Step 4: Model Training
    print("\n🤖 Step 4: Model Training")
    print("-" * 30)
    
    # Prepare features and target
    feature_columns = ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']
    available_features = [col for col in feature_columns if col in data_engineered.columns]
    
    # Add engineered features
    if 'PM25_PM10_Ratio' in data_engineered.columns:
        available_features.append('PM25_PM10_Ratio')
    if 'Combined_Pollution_Index' in data_engineered.columns:
        available_features.append('Combined_Pollution_Index')
    
    X = data_engineered[available_features]
    y = data_engineered['AirQuality_Category']
    
    print(f"📊 Training features: {available_features}")
    
    # Initialize and train model
    model = KNNAirQualityModel(task_type='classification')
    
    # Quick hyperparameter tuning (limited for demo)
    print("🔧 Performing hyperparameter tuning...")
    tuning_results = model.hyperparameter_tuning(X, y, cv_folds=3, scoring='accuracy')
    print(f"✅ Best parameters: {tuning_results['best_params']}")
    print(f"📊 Best CV score: {tuning_results['best_score']:.4f}")
    
    # Train final model
    print("🚀 Training final model...")
    training_results = model.train(X, y, test_size=0.2, random_state=42)
    
    print(f"✅ Training completed!")
    print(f"   Training Accuracy: {training_results['train_accuracy']:.4f}")
    print(f"   Test Accuracy: {training_results['test_accuracy']:.4f}")
    
    # Step 5: Model Evaluation
    print("\n📊 Step 5: Model Evaluation")
    print("-" * 30)
    
    # Cross-validation
    cv_results = model.cross_validate(X, y, cv_folds=5)
    accuracy_cv = cv_results.get('accuracy', {})
    print(f"🔄 Cross-validation accuracy: {accuracy_cv['mean']:.4f} (±{accuracy_cv['std']:.4f})")
    
    # Feature importance
    importance_data = model.feature_importance(X, y, method='permutation')
    print("\n🎯 Top 3 Most Important Features:")
    feature_names = importance_data['feature_names']
    importance_scores = importance_data['importance_mean']
    ranking = importance_data['ranking']
    
    for i in range(min(3, len(ranking))):
        idx = ranking[i]
        print(f"   {i+1}. {feature_names[idx]}: {importance_scores[idx]:.4f}")
    
    # Step 6: Making Predictions
    print("\n🔮 Step 6: Making Predictions")
    print("-" * 30)
    
    # Sample predictions
    sample_data = X.sample(n=5, random_state=42)
    predictions = model.predict(sample_data)
    predictions_with_conf, confidence_scores = model.predict_with_confidence(sample_data)
    
    print("📊 Sample Predictions:")
    print(f"{'Index':<8} {'Prediction':<12} {'Confidence':<12}")
    print("-" * 35)
    
    for i, (idx, pred, conf) in enumerate(zip(sample_data.index, predictions, confidence_scores)):
        print(f"{idx:<8} {pred:<12} {conf:<12.3f}")
    
    # Step 7: Visualization (optional)
    print("\n📊 Step 7: Creating Visualizations")
    print("-" * 30)
    
    try:
        visualizer = AirQualityVisualizer()
        
        # Create and save some plots
        import matplotlib.pyplot as plt
        plt.style.use('default')  # Ensure compatibility
        
        # Data overview
        fig1 = visualizer.plot_data_overview(data_engineered, available_features)
        if fig1:
            print("✅ Data overview plot created")
            plt.close(fig1)  # Close to save memory
        
        # Model performance
        fig2 = visualizer.plot_model_performance(training_results, 'classification')
        if fig2:
            print("✅ Model performance plot created")
            plt.close(fig2)
        
        # Feature importance
        fig3 = visualizer.plot_feature_importance(importance_data, top_k=len(available_features))
        if fig3:
            print("✅ Feature importance plot created")
            plt.close(fig3)
            
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
    
    # Step 8: Model Persistence
    print("\n💾 Step 8: Saving Results")
    print("-" * 30)
    
    try:
        import os
        output_dir = "example_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = f"{output_dir}/example_model.joblib"
        ModelPersistence.save_model(model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Save results summary
        summary_path = f"{output_dir}/example_summary.json"
        ModelPersistence.save_model_summary(model, training_results, summary_path)
        print(f"✅ Summary saved to {summary_path}")
        
        # Export data
        data_path = f"{output_dir}/processed_data.csv"
        ExportManager.export_to_csv(data_engineered, data_path)
        print(f"✅ Data exported to {data_path}")
        
        print(f"\n📁 All outputs saved to '{output_dir}' directory")
        
    except Exception as e:
        print(f"⚠️ Export failed: {e}")
    
    # Step 9: Interactive Prediction Function
    print("\n🔮 Step 9: Interactive Prediction Example")
    print("-" * 30)
    
    def predict_air_quality_interactive(pm10, pm25, so2, co, o3, no2):
        """Interactive prediction function"""
        import pandas as pd
        import numpy as np
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'PM10': [pm10],
            'PM25': [pm25], 
            'SO2': [so2],
            'CO': [co],
            'O3': [o3],
            'NO2': [no2]
        })
        
        # Add engineered features if model expects them
        if 'PM25_PM10_Ratio' in available_features:
            input_data['PM25_PM10_Ratio'] = input_data['PM25'] / (input_data['PM10'] + 1e-8)
        
        if 'Combined_Pollution_Index' in available_features:
            # Simple normalization for demo
            normalized = [pm10/100, pm25/50, so2/50, co/10, o3/100, no2/50]
            input_data['Combined_Pollution_Index'] = [np.mean(normalized)]
        
        # Select features in correct order
        input_data = input_data[available_features]
        
        # Make prediction
        prediction, confidence = model.predict_with_confidence(input_data)
        return prediction[0], confidence[0]
    
    # Example prediction
    example_pm10, example_pm25 = 75, 35
    example_so2, example_co = 20, 2.0
    example_o3, example_no2 = 85, 40
    
    pred_category, pred_confidence = predict_air_quality_interactive(
        example_pm10, example_pm25, example_so2, 
        example_co, example_o3, example_no2
    )
    
    print(f"📊 Example Input:")
    print(f"   PM10: {example_pm10} μg/m³, PM2.5: {example_pm25} μg/m³")
    print(f"   SO2: {example_so2} μg/m³, CO: {example_co} mg/m³") 
    print(f"   O3: {example_o3} μg/m³, NO2: {example_no2} μg/m³")
    
    print(f"\n🎯 Prediction:")
    print(f"   Air Quality: {pred_category}")
    print(f"   Confidence: {pred_confidence:.3f}")
    
    # Final Summary
    print(f"\n🎉 EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📊 Model Performance Summary:")
    print(f"   • Test Accuracy: {training_results['test_accuracy']:.1%}")
    print(f"   • CV Accuracy: {accuracy_cv['mean']:.1%} (±{accuracy_cv['std']:.1%})")
    print(f"   • Best K value: {tuning_results['best_params'].get('n_neighbors', 'N/A')}")
    print(f"   • Features used: {len(available_features)}")
    print(f"   • Training samples: {training_results['train_size']}")
    print(f"   • Test samples: {training_results['test_size']}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Use your own air quality data")
    print(f"   2. Experiment with different features")
    print(f"   3. Try different algorithms for comparison")
    print(f"   4. Deploy the model for real-time prediction")
    
    print(f"\n📚 For more details, check:")
    print(f"   • comprehensive_air_quality_analysis.ipynb")
    print(f"   • README.md")
    print(f"   • Generated output files")

if __name__ == "__main__":
    main()