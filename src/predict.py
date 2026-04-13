"""
Churn Prediction Script for Retail Customers
=========================================================================
This script provides functions to:
1. Load trained model and preprocessor
2. Predict churn probability for new customers
3. Batch prediction on CSV files
4. Feature importance analysis

Usage:
    # Single customer prediction
    python predict.py --customer-id 12345

    # Batch prediction from CSV
    python predict.py --input new_customers.csv --output predictions.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def load_model_artifacts(models_dir: Path) -> tuple:
    """Load the trained model, preprocessor, and feature names."""
    # Find the best model file
    model_files = list(models_dir.glob('best_model_*.joblib'))
    if not model_files:
        raise FileNotFoundError(f"No trained model found in {models_dir}")

    model_path = model_files[0]
    preprocessor_path = models_dir / 'preprocessor.joblib'
    features_path = models_dir / 'feature_names.json'

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading preprocessor from: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    if features_path.exists():
        with open(features_path) as f:
            feature_names = json.load(f)
    else:
        feature_names = None

    return model, preprocessor, feature_names


def prepare_customer_data(customer_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare customer data for prediction (remove leaky and ID columns)."""
    # Columns to exclude (same as training)
    leaky_features = ['ChurnRiskCategory', 'RFMSegment', 'CustomerType', 'Recency']
    id_columns = ['CustomerID', 'Churn']

    cols_to_drop = [c for c in leaky_features + id_columns if c in customer_data.columns]

    if cols_to_drop:
        print(f"Dropping columns: {cols_to_drop}")
        customer_data = customer_data.drop(columns=cols_to_drop)

    return customer_data


def predict_churn(customer_data: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    """Predict churn probability for customer data."""
    # Prepare data
    X = prepare_customer_data(customer_data.copy())

    # Transform using preprocessor
    X_processed = preprocessor.transform(X)

    # Predict
    churn_proba = model.predict_proba(X_processed)[:, 1]
    churn_pred = model.predict(X_processed)

    # Create results dataframe
    results = pd.DataFrame({
        'ChurnProbability': churn_proba,
        'ChurnPrediction': churn_pred,
        'RiskLevel': pd.cut(churn_proba,
                           bins=[0, 0.3, 0.6, 0.8, 1.0],
                           labels=['Low', 'Medium', 'High', 'Critical'])
    })

    # Add CustomerID if available
    if 'CustomerID' in customer_data.columns:
        results.insert(0, 'CustomerID', customer_data['CustomerID'].values)

    return results


def get_feature_importance(model, feature_names: list = None) -> pd.DataFrame:
    """Get feature importances from the model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        if feature_names and len(feature_names) == len(importances):
            names = feature_names
        else:
            names = [f'Feature_{i}' for i in range(len(importances))]

        importance_df = pd.DataFrame({
            'Feature': names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None


def predict_single_customer(customer_id: int, data_path: Path, model, preprocessor) -> dict:
    """Predict churn for a single customer by ID."""
    df = pd.read_csv(data_path)

    if customer_id not in df['CustomerID'].values:
        raise ValueError(f"Customer ID {customer_id} not found in data")

    customer_data = df[df['CustomerID'] == customer_id]
    results = predict_churn(customer_data, model, preprocessor)

    return {
        'CustomerID': customer_id,
        'ChurnProbability': float(results['ChurnProbability'].iloc[0]),
        'ChurnPrediction': int(results['ChurnPrediction'].iloc[0]),
        'RiskLevel': str(results['RiskLevel'].iloc[0])
    }


def batch_predict(input_path: Path, output_path: Path, model, preprocessor):
    """Run batch prediction on a CSV file."""
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Processing {len(df)} customers")

    results = predict_churn(df, model, preprocessor)

    # Merge with original data if CustomerID exists
    if 'CustomerID' in df.columns:
        output_df = df[['CustomerID']].join(results.drop('CustomerID', axis=1, errors='ignore'))
    else:
        output_df = results

    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total customers: {len(results)}")
    print(f"Predicted to churn: {results['ChurnPrediction'].sum()} ({results['ChurnPrediction'].mean()*100:.1f}%)")
    print(f"\nRisk Level Distribution:")
    print(results['RiskLevel'].value_counts())

    return output_df


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Predict customer churn')
    parser.add_argument('--customer-id', type=int, help='Predict for a specific customer ID')
    parser.add_argument('--input', type=str, help='Input CSV file for batch prediction')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file')
    parser.add_argument('--importance', action='store_true', help='Show feature importances')

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    processed_data_path = project_root / 'data' / 'processed' / 'retail_customers_cleaned.csv'

    # Load model artifacts
    print("=" * 50)
    print("LOADING MODEL ARTIFACTS")
    print("=" * 50)
    model, preprocessor, feature_names = load_model_artifacts(models_dir)
    print("Model loaded successfully!\n")

    # Show feature importance if requested
    if args.importance:
        print("=" * 50)
        print("FEATURE IMPORTANCE")
        print("=" * 50)
        importance_df = get_feature_importance(model, feature_names)
        if importance_df is not None:
            print(importance_df.head(20).to_string(index=False))
        return

    # Single customer prediction
    if args.customer_id:
        print("=" * 50)
        print(f"PREDICTING CHURN FOR CUSTOMER {args.customer_id}")
        print("=" * 50)
        result = predict_single_customer(args.customer_id, processed_data_path, model, preprocessor)
        print(f"\nResults:")
        print(f"  Customer ID: {result['CustomerID']}")
        print(f"  Churn Probability: {result['ChurnProbability']:.4f}")
        print(f"  Churn Prediction: {'Yes' if result['ChurnPrediction'] else 'No'}")
        print(f"  Risk Level: {result['RiskLevel']}")
        return

    # Batch prediction
    if args.input:
        print("=" * 50)
        print("BATCH PREDICTION")
        print("=" * 50)
        input_path = Path(args.input)
        output_path = Path(args.output)
        batch_predict(input_path, output_path, model, preprocessor)
        return

    # Default: show usage
    print("Usage examples:")
    print("  python predict.py --customer-id 12345")
    print("  python predict.py --input new_customers.csv --output predictions.csv")
    print("  python predict.py --importance")


if __name__ == '__main__':
    main()
