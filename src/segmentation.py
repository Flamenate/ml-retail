"""
Customer Segmentation Pipeline for Marketing Strategy
=========================================================================
This script performs customer segmentation using:
1. RFM Analysis (Recency, Frequency, Monetary)
2. K-Means Clustering
3. Cluster profiling for marketing insights

The resulting segments can be used to:
- Customize marketing campaigns
- Identify at-risk customers
- Target high-value customers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the preprocessed dataset."""
    print("=" * 60)
    print("STEP 1: Loading data for segmentation")
    print("=" * 60)
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} customers")
    return df


def prepare_rfm_features(df: pd.DataFrame) -> tuple:
    """Extract and prepare RFM features for clustering."""
    print("\n" + "=" * 60)
    print("STEP 2: Preparing RFM features")
    print("=" * 60)

    # Note: Even though Recency is leaky for churn prediction,
    # it's a valid feature for customer segmentation
    rfm_features = ['Recency', 'Frequency', 'MonetaryTotal']

    # Check if features exist
    missing_features = [f for f in rfm_features if f not in df.columns]
    if missing_features:
        # Load raw data if processed doesn't have Recency
        print(f"Missing features in processed data: {missing_features}")
        print("Loading original data for RFM features...")
        raw_path = Path(__file__).parent.parent / 'data' / 'raw' / 'retail_customers_COMPLETE_CATEGORICAL.csv'
        raw_df = pd.read_csv(raw_path)
        df = df.merge(raw_df[['CustomerID'] + missing_features], on='CustomerID', how='left')

    # Select RFM features
    rfm_data = df[['CustomerID'] + rfm_features].copy()

    print(f"RFM features statistics:")
    print(rfm_data[rfm_features].describe())

    return rfm_data, rfm_features


def scale_features(rfm_data: pd.DataFrame, rfm_features: list) -> tuple:
    """Scale RFM features for clustering."""
    print("\n" + "=" * 60)
    print("STEP 3: Scaling features")
    print("=" * 60)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data[rfm_features])

    print("Features scaled using StandardScaler")

    return rfm_scaled, scaler


def find_optimal_clusters(rfm_scaled: np.ndarray, max_clusters: int = 10) -> int:
    """Find optimal number of clusters using silhouette score."""
    print("\n" + "=" * 60)
    print("STEP 4: Finding optimal number of clusters")
    print("=" * 60)

    silhouette_scores = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        sil_score = silhouette_score(rfm_scaled, kmeans.labels_)
        db_score = davies_bouldin_score(rfm_scaled, kmeans.labels_)
        silhouette_scores.append(sil_score)
        print(f"  k={k}: Silhouette={sil_score:.4f}, DB={db_score:.4f}")

    # Find optimal k based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal clusters based on Silhouette score: {optimal_k}")

    return optimal_k


def perform_clustering(rfm_scaled: np.ndarray, n_clusters: int) -> tuple:
    """Perform K-Means clustering."""
    print("\n" + "=" * 60)
    print(f"STEP 5: Performing K-Means clustering (k={n_clusters})")
    print("=" * 60)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(rfm_scaled)

    print(f"Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} customers ({count/len(cluster_labels)*100:.1f}%)")

    return kmeans, cluster_labels


def profile_clusters(rfm_data: pd.DataFrame, cluster_labels: np.ndarray,
                     rfm_features: list, df_full: pd.DataFrame) -> pd.DataFrame:
    """Create detailed cluster profiles."""
    print("\n" + "=" * 60)
    print("STEP 6: Profiling clusters")
    print("=" * 60)

    rfm_data = rfm_data.copy()
    rfm_data['Cluster'] = cluster_labels

    # Merge with full data to get churn info
    rfm_data = rfm_data.merge(df_full[['CustomerID', 'Churn']], on='CustomerID', how='left')

    # Calculate cluster profiles
    profiles = rfm_data.groupby('Cluster').agg({
        'Recency': ['mean', 'median'],
        'Frequency': ['mean', 'median'],
        'MonetaryTotal': ['mean', 'median', 'sum'],
        'Churn': ['mean', 'sum'],
        'CustomerID': 'count'
    }).round(2)

    profiles.columns = ['_'.join(col).strip() for col in profiles.columns.values]
    profiles = profiles.rename(columns={
        'CustomerID_count': 'CustomerCount',
        'Churn_mean': 'ChurnRate',
        'Churn_sum': 'ChurnedCustomers'
    })

    # Name clusters based on RFM characteristics
    cluster_names = []
    for cluster in profiles.index:
        R_mean = profiles.loc[cluster, 'Recency_mean']
        F_mean = profiles.loc[cluster, 'Frequency_mean']
        M_mean = profiles.loc[cluster, 'MonetaryTotal_mean']

        # Determine cluster name based on RFM values
        R_median = rfm_data['Recency'].median()
        F_median = rfm_data['Frequency'].median()
        M_median = rfm_data['MonetaryTotal'].median()

        if R_mean < R_median and F_mean > F_median and M_mean > M_median:
            name = "Champions"
        elif R_mean < R_median and F_mean > F_median:
            name = "Loyal Customers"
        elif R_mean < R_median and M_mean > M_median:
            name = "Potential Loyalists"
        elif R_mean > R_median * 1.5:
            name = "At Risk / Churning"
        elif F_mean < F_median and M_mean < M_median:
            name = "New / Low-Value"
        else:
            name = "Average Customers"

        cluster_names.append(name)

    profiles['ClusterName'] = cluster_names

    print("\nCluster Profiles:")
    print(profiles.to_string())

    return profiles, rfm_data


def plot_segmentation_results(rfm_data: pd.DataFrame, profiles: pd.DataFrame,
                               output_dir: Path):
    """Generate and save segmentation visualizations."""
    print("\n" + "=" * 60)
    print("STEP 7: Generating visualizations")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cluster distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart of customer distribution
    cluster_counts = rfm_data['Cluster'].value_counts().sort_index()
    labels = [f"Cluster {i}\n({profiles.loc[i, 'ClusterName']})" for i in cluster_counts.index]
    axes[0].pie(cluster_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Customer Distribution by Segment')

    # Bar chart of churn rate by cluster
    churn_rates = rfm_data.groupby('Cluster')['Churn'].mean() * 100
    bars = axes[1].bar(churn_rates.index, churn_rates.values, color='coral')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Churn Rate (%)')
    axes[1].set_title('Churn Rate by Segment')
    for bar, rate in zip(bars, churn_rates.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'segment_distribution.png', dpi=150)
    plt.close()

    # 2. Cluster profile heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize RFM values for heatmap
    rfm_means = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryTotal']].mean()
    rfm_normalized = (rfm_means - rfm_means.min()) / (rfm_means.max() - rfm_means.min())

    sns.heatmap(rfm_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn_r',
                xticklabels=[f"Cluster {i}\n({profiles.loc[i, 'ClusterName']})"
                            for i in rfm_normalized.index],
                yticklabels=['Recency\n(lower=better)', 'Frequency\n(higher=better)',
                            'Monetary\n(higher=better)'],
                ax=ax)
    ax.set_title('Normalized RFM Profile by Segment')

    plt.tight_layout()
    plt.savefig(output_dir / 'segment_heatmap.png', dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def generate_marketing_recommendations(profiles: pd.DataFrame) -> dict:
    """Generate marketing recommendations for each segment."""
    print("\n" + "=" * 60)
    print("STEP 8: Generating marketing recommendations")
    print("=" * 60)

    recommendations = {}

    for cluster in profiles.index:
        name = profiles.loc[cluster, 'ClusterName']
        churn_rate = profiles.loc[cluster, 'ChurnRate']
        recency = profiles.loc[cluster, 'Recency_mean']
        frequency = profiles.loc[cluster, 'Frequency_mean']
        monetary = profiles.loc[cluster, 'MonetaryTotal_mean']

        if name == "Champions":
            rec = {
                "priority": "HIGH VALUE",
                "actions": [
                    "Reward with exclusive offers and early access to new products",
                    "Implement loyalty program with VIP benefits",
                    "Request reviews and referrals",
                    "Personalized thank-you communications"
                ],
                "channels": ["Email", "Direct Mail", "Personal Account Manager"]
            }
        elif name == "Loyal Customers":
            rec = {
                "priority": "HIGH",
                "actions": [
                    "Upsell higher-value products",
                    "Cross-sell complementary items",
                    "Exclusive previews and early access",
                    "Loyalty points multiplier events"
                ],
                "channels": ["Email", "App Notifications", "SMS"]
            }
        elif name == "Potential Loyalists":
            rec = {
                "priority": "MEDIUM-HIGH",
                "actions": [
                    "Nurture with membership benefits",
                    "Recommend products based on purchase history",
                    "Offer free shipping thresholds to increase basket size",
                    "Early access to sales"
                ],
                "channels": ["Email", "Retargeting Ads"]
            }
        elif name == "At Risk / Churning":
            rec = {
                "priority": "URGENT",
                "actions": [
                    "Win-back campaign with special discounts",
                    "Survey to understand reasons for inactivity",
                    "Personalized 'We miss you' messaging",
                    "Time-limited exclusive offers"
                ],
                "channels": ["Email", "SMS", "Direct Mail"]
            }
        elif name == "New / Low-Value":
            rec = {
                "priority": "MEDIUM",
                "actions": [
                    "Welcome series with onboarding content",
                    "First-purchase incentive if not converted",
                    "Product education and guides",
                    "Social proof (reviews, testimonials)"
                ],
                "channels": ["Email", "Social Media", "Display Ads"]
            }
        else:
            rec = {
                "priority": "STANDARD",
                "actions": [
                    "Regular promotional communications",
                    "Seasonal offers",
                    "Product recommendations based on browsing",
                    "Newsletter engagement"
                ],
                "channels": ["Email", "Social Media"]
            }

        recommendations[f"Cluster_{cluster}_{name}"] = rec
        print(f"\nCluster {cluster} ({name}) - Priority: {rec['priority']}")
        print(f"  Churn Rate: {churn_rate*100:.1f}%")
        print(f"  Actions: {rec['actions'][0]}...")

    return recommendations


def save_artifacts(kmeans: KMeans, scaler: StandardScaler, profiles: pd.DataFrame,
                   rfm_data: pd.DataFrame, recommendations: dict, output_dir: Path,
                   data_processed_dir: Path):
    """Save clustering artifacts."""
    print("\n" + "=" * 60)
    print("STEP 9: Saving artifacts")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)

    # Save KMeans model
    joblib.dump(kmeans, output_dir / 'kmeans_segmentation.joblib')
    print(f"Saved KMeans model to {output_dir / 'kmeans_segmentation.joblib'}")

    # Save scaler
    joblib.dump(scaler, output_dir / 'rfm_scaler.joblib')
    print(f"Saved RFM scaler to {output_dir / 'rfm_scaler.joblib'}")

    # Save profiles
    profiles.to_csv(output_dir / 'segment_profiles.csv')
    print(f"Saved profiles to {output_dir / 'segment_profiles.csv'}")

    # Save customer segments
    rfm_data.to_csv(data_processed_dir / 'customer_segments.csv', index=False)
    print(f"Saved customer segments to {data_processed_dir / 'customer_segments.csv'}")

    # Save recommendations
    with open(output_dir / 'marketing_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"Saved recommendations to {output_dir / 'marketing_recommendations.json'}")


def main():
    """Run the full segmentation pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    processed_data_path = project_root / 'data' / 'processed' / 'retail_customers_cleaned.csv'
    data_processed_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    reports_dir = project_root / 'reports'

    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # Load data
    df = load_data(processed_data_path)

    # Prepare RFM features (load from raw data since Recency was removed from processed)
    raw_path = project_root / 'data' / 'raw' / 'retail_customers_COMPLETE_CATEGORICAL.csv'
    raw_df = pd.read_csv(raw_path)
    rfm_features = ['Recency', 'Frequency', 'MonetaryTotal']

    # Merge RFM features with processed data
    rfm_data = df[['CustomerID']].merge(
        raw_df[['CustomerID'] + rfm_features],
        on='CustomerID',
        how='left'
    )

    print(f"\nRFM features prepared for {len(rfm_data)} customers")

    # Scale features
    rfm_scaled, scaler = scale_features(rfm_data, rfm_features)

    # Find optimal clusters
    optimal_k = find_optimal_clusters(rfm_scaled)

    # Use 4 clusters as a reasonable default if optimal is too low/high
    n_clusters = max(3, min(optimal_k, 5))
    print(f"\nUsing {n_clusters} clusters for segmentation")

    # Perform clustering
    kmeans, cluster_labels = perform_clustering(rfm_scaled, n_clusters)

    # Profile clusters
    profiles, rfm_data_with_clusters = profile_clusters(rfm_data, cluster_labels, rfm_features, df)

    # Generate visualizations
    plot_segmentation_results(rfm_data_with_clusters, profiles, reports_dir)

    # Generate marketing recommendations
    recommendations = generate_marketing_recommendations(profiles)

    # Save artifacts
    save_artifacts(kmeans, scaler, profiles, rfm_data_with_clusters, recommendations, models_dir, data_processed_dir)

    print("\n" + "=" * 60)
    print("SEGMENTATION COMPLETE")
    print("=" * 60)

    return kmeans, profiles, recommendations


if __name__ == '__main__':
    main()
