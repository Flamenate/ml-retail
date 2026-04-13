"""
Data Preprocessing Pipeline for Retail Customer Churn Prediction
=========================================================================
This script handles:
1. Loading raw data
2. Missing value imputation (Age with KNN)
3. Outlier detection and correction (SupportTickets, Satisfaction)
4. Date parsing and feature extraction (RegistrationDate)
5. IP feature engineering (IsPrivateIP, LoginCountry)
6. Saving processed data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
import sys

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils import is_private_ip, load_dbip, lookup_country, DBIP_CSV_PATH


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Load raw dataset."""
    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Target distribution (Churn):\n{df['Churn'].value_counts()}")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not useful for modeling."""
    print("\n" + "=" * 60)
    print("STEP 2: Dropping unnecessary columns")
    print("=" * 60)

    # NewsletterSubscribed is mentioned as unreliable in the project
    cols_to_drop = ['NewsletterSubscribed']
    existing_cols = [c for c in cols_to_drop if c in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        print(f"Dropped columns: {existing_cols}")

    return df


def impute_missing_age(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing Age values using KNN imputation."""
    print("\n" + "=" * 60)
    print("STEP 3: Imputing missing Age values (KNN)")
    print("=" * 60)

    missing_before = df['Age'].isna().sum()
    print(f"Missing Age values before: {missing_before} ({missing_before/len(df)*100:.2f}%)")

    if missing_before == 0:
        print("No missing Age values - skipping imputation")
        return df

    # KNN Imputation using correlated features
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')

    numeric_features = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg',
                        'CustomerTenureDays', 'FirstPurchaseDaysAgo', 'Age']

    # Only use features that exist
    available_features = [f for f in numeric_features if f in df.columns]

    df[available_features] = knn_imputer.fit_transform(df[available_features])

    # Clip Age to realistic bounds (18-81 per project specs)
    df['Age'] = df['Age'].clip(lower=18, upper=81)

    print(f"Missing Age values after: {df['Age'].isna().sum()}")
    print(f"Age stats: Mean={df['Age'].mean():.2f}, Median={df['Age'].median():.2f}, Std={df['Age'].std():.2f}")

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Handle sentinel values in SupportTickets and Satisfaction."""
    print("\n" + "=" * 60)
    print("STEP 4: Handling outliers (sentinel values)")
    print("=" * 60)

    # SupportTicketsCount: valid range 0-15, outliers: -1 and 999
    support_outlier_mask = df['SupportTicketsCount'].isin([-1, 999])
    print(f"SupportTicketsCount outliers: {support_outlier_mask.sum()} ({support_outlier_mask.sum()/len(df)*100:.2f}%)")

    # SatisfactionScore: valid range 1-5, outliers: -1, 0, and 99
    satisfaction_outlier_mask = df['SatisfactionScore'].isin([-1, 0, 99])
    print(f"SatisfactionScore outliers: {satisfaction_outlier_mask.sum()} ({satisfaction_outlier_mask.sum()/len(df)*100:.2f}%)")

    # Replace outliers with NaN
    df.loc[support_outlier_mask, 'SupportTicketsCount'] = np.nan
    df.loc[satisfaction_outlier_mask, 'SatisfactionScore'] = np.nan

    # Impute with median
    df['SupportTicketsCount'] = df['SupportTicketsCount'].fillna(df['SupportTicketsCount'].median())
    df['SatisfactionScore'] = df['SatisfactionScore'].fillna(df['SatisfactionScore'].median())

    # Ensure valid ranges
    df['SupportTicketsCount'] = df['SupportTicketsCount'].clip(lower=0, upper=15)
    df['SatisfactionScore'] = df['SatisfactionScore'].clip(lower=1, upper=5)

    print(f"SupportTicketsCount stats: Mean={df['SupportTicketsCount'].mean():.2f}, Median={df['SupportTicketsCount'].median():.2f}")
    print(f"SatisfactionScore stats: Mean={df['SatisfactionScore'].mean():.2f}, Median={df['SatisfactionScore'].median():.2f}")

    return df


def impute_remaining_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute any remaining missing values in numeric columns."""
    print("\n" + "=" * 60)
    print("STEP 4b: Imputing remaining missing values")
    print("=" * 60)

    # Check for remaining missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_counts = df[numeric_cols].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if len(cols_with_missing) == 0:
        print("No remaining missing values in numeric columns")
        return df

    print(f"Columns with missing values: {cols_with_missing.to_dict()}")

    # Impute with median for each column
    for col in cols_with_missing.index:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  {col}: imputed {cols_with_missing[col]} values with median={median_val:.2f}")

    return df


def parse_registration_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse RegistrationDate and extract temporal features."""
    print("\n" + "=" * 60)
    print("STEP 5: Parsing RegistrationDate")
    print("=" * 60)

    # Parse with UK date format priority (dayfirst=True)
    df['RegistrationDate'] = pd.to_datetime(
        df['RegistrationDate'],
        dayfirst=True,
        errors='coerce'
    )

    nat_count = df['RegistrationDate'].isna().sum()
    print(f"Unparseable entries (NaT): {nat_count}")

    # Extract temporal features
    df['RegYear'] = df['RegistrationDate'].dt.year
    df['RegMonth'] = df['RegistrationDate'].dt.month
    df['RegDay'] = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday

    # Drop the original raw column
    df = df.drop(columns=['RegistrationDate'])

    print("Extracted features: RegYear, RegMonth, RegDay, RegWeekday")

    return df


def engineer_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from LastLoginIP."""
    print("\n" + "=" * 60)
    print("STEP 6: Engineering IP features")
    print("=" * 60)

    # IsPrivateIP feature
    df['IsPrivateIP'] = df['LastLoginIP'].apply(is_private_ip)

    # LoginCountry feature (requires DB-IP Lite CSV)
    if DBIP_CSV_PATH.exists():
        print(f"Loading DB-IP database from {DBIP_CSV_PATH}")
        db = load_dbip(DBIP_CSV_PATH)
        starts = db['ip_start_int'].tolist()
        ends = db['ip_end_int'].tolist()
        countries = db['country'].tolist()
        df['LoginCountry'] = df['LastLoginIP'].apply(
            lambda ip: lookup_country(ip, starts, ends, countries)
        )
        print(f"LoginCountry distribution (top 5):\n{df['LoginCountry'].value_counts().head()}")
    else:
        df['LoginCountry'] = 'Unknown'
        print(f"[WARNING] DB-IP file not found at {DBIP_CSV_PATH}")
        print("LoginCountry set to 'Unknown' for all rows.")

    # Drop raw IP column
    df = df.drop(columns=['LastLoginIP'])

    print(f"IsPrivateIP distribution:\n{df['IsPrivateIP'].value_counts()}")

    return df


def save_processed_data(df: pd.DataFrame, output_path: Path):
    """Save processed data to CSV."""
    print("\n" + "=" * 60)
    print("STEP 7: Saving processed data")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    print(f"Final shape: {df.shape}")


def main():
    """Run the full preprocessing pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / 'data' / 'raw' / 'retail_customers_COMPLETE_CATEGORICAL.csv'
    processed_data_path = project_root / 'data' / 'processed' / 'retail_customers_cleaned.csv'

    print("=" * 60)
    print("RETAIL CUSTOMER DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Run pipeline
    df = load_raw_data(raw_data_path)
    df = drop_unnecessary_columns(df)
    df = impute_missing_age(df)
    df = handle_outliers(df)
    df = impute_remaining_missing(df)
    df = parse_registration_date(df)
    df = engineer_ip_features(df)
    save_processed_data(df, processed_data_path)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    return df


if __name__ == '__main__':
    main()
