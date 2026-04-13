"""
Model Training Pipeline for Retail Customer Churn Prediction
=========================================================================
This script handles:
1. Loading processed data
2. Encoding categorical variables (Label + One-Hot)
3. Feature scaling with StandardScaler
4. Train/test split (stratified)
5. Handling class imbalance with SMOTE
6. Training multiple classification models
7. Hyperparameter tuning with GridSearchCV
8. Model evaluation and comparison
9. Saving models and artifacts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


def load_processed_data(data_path: Path) -> pd.DataFrame:
    """Load the preprocessed dataset."""
    print("=" * 60)
    print("STEP 1: Loading processed data")
    print("=" * 60)
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Target distribution (Churn):\n{df['Churn'].value_counts()}")
    print(f"Churn rate: {df['Churn'].mean()*100:.2f}%")
    return df


def prepare_features(df: pd.DataFrame):
    """Separate features and target, identify column types."""
    print("\n" + "=" * 60)
    print("STEP 2: Preparing features")
    print("=" * 60)

    # Drop CustomerID (identifier, not predictive)
    # Drop leaky features that are derived from or contain churn information:
    # - ChurnRiskCategory: directly derived from Churn target
    # - RFMSegment: "Dormants" category = 100% churned (derived from RFM which includes churn behavior)
    # - CustomerType: "Perdu" category = 100% churned (lost customer = churned)
    # - Recency: CRITICAL - perfectly separates churn (Churn=1 when Recency>=91 days)
    #   This is likely the definition of churn itself (inactive > 90 days = churned)
    leaky_features = ['ChurnRiskCategory', 'RFMSegment', 'CustomerType', 'Recency']
    X = df.drop(columns=['Churn', 'CustomerID'] + leaky_features)
    y = df['Churn']

    print(f"Dropped leaky features: {leaky_features}")

    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Categorical columns: {categorical_cols}")

    return X, y, numeric_cols, categorical_cols


def create_preprocessor(numeric_cols: list, categorical_cols: list):
    """Create a preprocessing pipeline with scaling and encoding."""
    print("\n" + "=" * 60)
    print("STEP 3: Creating preprocessor")
    print("=" * 60)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    print("Preprocessor created with StandardScaler + OneHotEncoder")
    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split."""
    print("\n" + "=" * 60)
    print("STEP 4: Splitting data (stratified)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Train Churn rate: {y_train.mean()*100:.2f}%")
    print(f"Test Churn rate: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """Apply SMOTE to handle class imbalance."""
    print("\n" + "=" * 60)
    print("STEP 5: Applying SMOTE for class imbalance")
    print("=" * 60)

    print(f"Before SMOTE - Class distribution: {np.bincount(y_train.astype(int))}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE - Class distribution: {np.bincount(y_resampled.astype(int))}")

    return X_resampled, y_resampled


def get_models():
    """Return a dictionary of models to train."""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42)
    }
    return models


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }

    return metrics


def train_and_evaluate_models(X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray) -> tuple:
    """Train all models and evaluate them."""
    print("\n" + "=" * 60)
    print("STEP 6: Training and evaluating models")
    print("=" * 60)

    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print(f"  Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f}, ROC-AUC: {metrics['ROC_AUC']:.4f}")

    return trained_models, pd.DataFrame(results)


def tune_best_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                    y_test: np.ndarray, results_df: pd.DataFrame) -> tuple:
    """Perform hyperparameter tuning on the best performing model."""
    print("\n" + "=" * 60)
    print("STEP 7: Hyperparameter tuning (best model)")
    print("=" * 60)

    # Find best model based on F1 score
    best_model_name = results_df.loc[results_df['F1'].idxmax(), 'Model']
    print(f"Best model based on F1: {best_model_name}")

    # Define parameter grids for different models
    param_grids = {
        'DecisionTree': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'min_samples_split': [2, 5]
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2']
        }
    }

    if best_model_name in param_grids:
        model_class = get_models()[best_model_name].__class__
        param_grid = param_grids[best_model_name]

        print(f"Running GridSearchCV with {len(param_grid)} parameters...")

        grid_search = GridSearchCV(
            model_class(random_state=42) if hasattr(model_class(), 'random_state') else model_class(),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        # Evaluate tuned model on test set
        tuned_metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test, f"{best_model_name}_Tuned")
        print(f"Tuned model test F1: {tuned_metrics['F1']:.4f}, ROC-AUC: {tuned_metrics['ROC_AUC']:.4f}")

        return grid_search.best_estimator_, tuned_metrics, best_model_name
    else:
        print(f"No parameter grid defined for {best_model_name}, skipping tuning")
        return None, None, best_model_name


def plot_results(results_df: pd.DataFrame, trained_models: dict, X_test: np.ndarray,
                 y_test: np.ndarray, output_dir: Path):
    """Generate and save evaluation plots."""
    print("\n" + "=" * 60)
    print("STEP 8: Generating evaluation plots")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
    results_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Model Comparison - Classification Metrics')
    axes[0].set_ylabel('Score')
    axes[0].legend(loc='lower right')
    axes[0].tick_params(axis='x', rotation=45)

    # ROC-AUC comparison
    results_df.set_index('Model')['ROC_AUC'].plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title('Model Comparison - ROC-AUC')
    axes[1].set_ylabel('ROC-AUC Score')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150)
    plt.close()

    # 2. ROC curves
    plt.figure(figsize=(10, 8))
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150)
    plt.close()

    # 3. Confusion matrix for best model
    best_model_name = results_df.loc[results_df['F1'].idxmax(), 'Model']
    best_model = trained_models[best_model_name]
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_best.png', dpi=150)
    plt.close()

    # 4. Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [f'Feature_{i}' for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importances - {best_model_name}')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")


def save_artifacts(preprocessor, best_model, model_name: str, results_df: pd.DataFrame,
                   feature_names: list, output_dir: Path, train_test_dir: Path,
                   X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """Save models, preprocessor, and other artifacts."""
    print("\n" + "=" * 60)
    print("STEP 9: Saving artifacts")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_test_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessor
    joblib.dump(preprocessor, output_dir / 'preprocessor.joblib')
    print(f"Saved preprocessor to {output_dir / 'preprocessor.joblib'}")

    # Save best model
    model_path = output_dir / f'best_model_{model_name.lower()}.joblib'
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

    # Save results
    results_df.to_csv(output_dir / 'model_results.csv', index=False)
    print(f"Saved results to {output_dir / 'model_results.csv'}")

    # Save feature names
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print(f"Saved feature names to {output_dir / 'feature_names.json'}")

    # Save train/test data
    X_train.to_csv(train_test_dir / 'X_train.csv', index=False)
    X_test.to_csv(train_test_dir / 'X_test.csv', index=False)
    y_train.to_csv(train_test_dir / 'y_train.csv', index=False)
    y_test.to_csv(train_test_dir / 'y_test.csv', index=False)
    print(f"Saved train/test splits to {train_test_dir}")


def main():
    """Run the full training pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    processed_data_path = project_root / 'data' / 'processed' / 'retail_customers_cleaned.csv'
    models_dir = project_root / 'models'
    reports_dir = project_root / 'reports'
    train_test_dir = project_root / 'data' / 'train_test'

    print("=" * 60)
    print("RETAIL CUSTOMER CHURN - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load data
    df = load_processed_data(processed_data_path)

    # Prepare features
    X, y, numeric_cols, categorical_cols = prepare_features(df)

    # Create preprocessor
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Fit preprocessor on training data and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after transformation
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names

    # Apply SMOTE on training data
    X_train_resampled, y_train_resampled = apply_smote(X_train_processed, y_train.values)

    # Train and evaluate models
    trained_models, results_df = train_and_evaluate_models(
        X_train_resampled, X_test_processed,
        y_train_resampled, y_test.values
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Tune best model
    tuned_model, tuned_metrics, best_model_name = tune_best_model(
        X_train_resampled, y_train_resampled,
        X_test_processed, y_test.values,
        results_df
    )

    # Use tuned model if available, otherwise use best from initial training
    final_model = tuned_model if tuned_model is not None else trained_models[best_model_name]
    final_model_name = f"{best_model_name}_Tuned" if tuned_model else best_model_name

    # Add tuned results to dataframe
    if tuned_metrics:
        results_df = pd.concat([results_df, pd.DataFrame([tuned_metrics])], ignore_index=True)

    # Generate plots
    plot_results(results_df, trained_models, X_test_processed, y_test.values, reports_dir)

    # Save all artifacts
    save_artifacts(
        preprocessor, final_model, final_model_name, results_df,
        all_feature_names, models_dir, train_test_dir,
        X_train, X_test, y_train, y_test
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest model: {final_model_name}")
    best_row = results_df[results_df['Model'] == final_model_name].iloc[0]
    print(f"  Accuracy: {best_row['Accuracy']:.4f}")
    print(f"  Precision: {best_row['Precision']:.4f}")
    print(f"  Recall: {best_row['Recall']:.4f}")
    print(f"  F1: {best_row['F1']:.4f}")
    print(f"  ROC-AUC: {best_row['ROC_AUC']:.4f}")

    return final_model, preprocessor, results_df


if __name__ == '__main__':
    main()
