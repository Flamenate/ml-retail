# Machine Learning Retail - Customer Churn Prediction

## Project Overview

This project implements a machine learning solution for a gift e-commerce shop to:
- **Customize marketing strategy** through customer segmentation
- **Reduce customer churn** through predictive modeling
- **Increase revenue** by targeting at-risk customers

## Dataset

The dataset `retail_customers_COMPLETE_CATEGORICAL.csv` contains **4,372 customers** with **52 features** including:
- **RFM metrics**: Recency, Frequency, Monetary values
- **Customer demographics**: Age, Gender, Country
- **Purchase behavior**: Preferred day/time, basket size, product diversity
- **Account information**: Registration date, account status, support tickets

## Project Structure

```
ml-retail/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── retail_customers_COMPLETE_CATEGORICAL.csv
│   ├── processed/                    # Cleaned and engineered data
│   │   ├── retail_customers_cleaned.csv
│   │   └── customer_segments.csv
│   └── train_test/                   # Train/test splits
│       ├── X_train.csv, X_test.csv
│       └── y_train.csv, y_test.csv
├── models/                           # Trained models and artifacts
│   ├── best_model_xgboost_tuned.joblib
│   ├── preprocessor.joblib
│   ├── kmeans_segmentation.joblib
│   └── feature_names.json
├── reports/                          # Visualizations and reports
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── confusion_matrix_best.png
│   ├── feature_importance.png
│   ├── segment_distribution.png
│   └── segment_heatmap.png
└── src/                              # Source code
    ├── preprocessing.py              # Data cleaning and feature engineering
    ├── train_model.py                # Model training pipeline
    ├── segmentation.py               # Customer segmentation (KMeans)
    ├── predict.py                    # Inference script
    └── utils.py                      # Utility functions
```

## Data Preprocessing

The preprocessing pipeline (`src/preprocessing.py`) handles:

1. **Missing Values**
   - Age: KNN imputation using 5 neighbors with distance weighting
   - AvgDaysBetweenPurchases: Median imputation (79 values)

2. **Outlier Treatment**
   - SupportTicketsCount: Replaced sentinel values (-1, 999) with median
   - SatisfactionScore: Replaced sentinel values (-1, 0, 99) with median

3. **Feature Engineering**
   - RegistrationDate: Extracted RegYear, RegMonth, RegDay, RegWeekday
   - LastLoginIP: Created IsPrivateIP and LoginCountry features

4. **Data Leakage Prevention**
   - Removed ChurnRiskCategory (directly derived from target)
   - Removed RFMSegment ("Dormants" = 100% churned)
   - Removed CustomerType ("Perdu" = 100% churned)
   - Removed Recency (defines churn: >90 days = churned)

## Model Training

The training pipeline (`src/train_model.py`) implements:

### Feature Processing
- StandardScaler for numeric features (35 features)
- OneHotEncoder for categorical features (14 features)
- SMOTE for class imbalance handling (33.26% churn rate)

### Models Evaluated
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost (Tuned)** | **99.1%** | **99.0%** | **98.3%** | **98.6%** | **99.97%** |
| Logistic Regression | 98.4% | 98.6% | 96.6% | 97.6% | 99.8% |
| Gradient Boosting | 98.3% | 96.9% | 97.9% | 97.4% | 99.9% |
| Decision Tree | 97.3% | 94.9% | 96.9% | 95.9% | 97.2% |
| SVM | 97.1% | 98.9% | 92.4% | 95.6% | 99.7% |
| Random Forest | 94.9% | 96.6% | 87.6% | 91.9% | 98.7% |
| KNN | 84.1% | 70.9% | 88.7% | 78.8% | 92.5% |

### Best Model: XGBoost
- **Hyperparameters**: learning_rate=0.1, max_depth=7, n_estimators=100, subsample=0.8
- **Cross-validation F1**: 99.09%

## Customer Segmentation

The segmentation pipeline (`src/segmentation.py`) uses KMeans clustering on RFM features:

### Segments Identified
| Cluster | Name | Customers | Churn Rate | Avg Recency | Avg Frequency | Avg Monetary |
|---------|------|-----------|------------|-------------|---------------|--------------|
| 0 | At Risk/Churning | 1,109 (25.4%) | 100% | 246 days | 1.85 | $460 |
| 1 | Champions | 26 (0.6%) | 0% | 6 days | 83.35 | $75,966 |
| 2 | Active Customers | 3,237 (74.0%) | 11% | 40 days | 5.55 | $1,796 |

### Marketing Recommendations
- **Champions**: VIP treatment, loyalty rewards, referral programs
- **Active Customers**: Upselling, cross-selling, engagement campaigns
- **At Risk**: Win-back campaigns, personalized discounts, surveys

## Usage

### Run Preprocessing
```bash
python src/preprocessing.py
```

### Train Models
```bash
python src/train_model.py
```

### Run Customer Segmentation
```bash
python src/segmentation.py
```

### Make Predictions
```bash
# Single customer prediction
python src/predict.py --customer-id 17850

# Batch prediction
python src/predict.py --input new_customers.csv --output predictions.csv

# View feature importance
python src/predict.py --importance
```

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
joblib
```

## Key Findings

1. **Feature Importance**: Loyalty level, favorite season, and preferred month are the strongest predictors of churn
2. **Churn Definition**: Customers with >90 days since last purchase are considered churned
3. **Class Imbalance**: 33.26% churn rate, handled with SMOTE
4. **Data Leakage**: Several features were derived from churn status and had to be excluded

## Generated Visualizations

The project generates six key visualizations in the `reports/` directory. Each graph provides critical insights for understanding model performance and customer behavior.

### 1. Model Comparison (`model_comparison.png`)

This visualization consists of two side-by-side bar charts comparing all trained models:

**Left Panel - Classification Metrics:**
- Displays four key metrics (Accuracy, Precision, Recall, F1 Score) for each of the 7 models
- Each model has 4 colored bars representing each metric
- Allows quick visual comparison to identify which models perform best overall
- XGBoost shows the highest bars across all metrics, confirming it as the best performer

**Right Panel - ROC-AUC Comparison:**
- Shows a single bar per model representing the Area Under the ROC Curve
- ROC-AUC measures the model's ability to distinguish between churned and non-churned customers
- Values closer to 1.0 indicate better discrimination capability
- Most models achieve >0.97 ROC-AUC, with XGBoost reaching 0.9997

**Business Insight:** This chart helps stakeholders understand relative model performance without diving into technical details. The consistently high scores across ensemble methods (XGBoost, GradientBoosting) justify our model selection.

---

### 2. ROC Curves (`roc_curves.png`)

This graph displays Receiver Operating Characteristic (ROC) curves for all models on a single plot:

**What it Shows:**
- X-axis: False Positive Rate (FPR) - proportion of non-churned customers incorrectly predicted as churned
- Y-axis: True Positive Rate (TPR) / Recall - proportion of churned customers correctly identified
- Each colored line represents one model's performance across all classification thresholds
- The diagonal dashed line represents random guessing (AUC = 0.5)

**How to Interpret:**
- Curves closer to the top-left corner indicate better performance
- The area under each curve (AUC) is shown in the legend
- A perfect classifier would have an AUC of 1.0 (curve hugging the top-left corner)

**Key Observations:**
- All models significantly outperform random guessing
- XGBoost, GradientBoosting, and Logistic Regression curves nearly overlap at the top-left
- KNN shows the most deviation, with a curve closer to the diagonal

**Business Insight:** ROC curves help determine the optimal threshold for classifying customers as "at risk." By adjusting this threshold, the business can balance between catching more potential churners (higher recall) vs. avoiding false alarms (higher precision).

---

### 3. Confusion Matrix (`confusion_matrix_best.png`)

A heatmap showing the prediction results of the best model (XGBoost) on the test set:

**Matrix Structure:**
```
                    Predicted
                 Not Churned | Churned
Actual  Not Churned    TN    |    FP
        Churned        FN    |    TP
```

**Cell Definitions:**
- **True Negatives (TN):** Customers correctly predicted as not churned
- **True Positives (TP):** Customers correctly predicted as churned
- **False Positives (FP):** Non-churned customers incorrectly flagged as churned (Type I error)
- **False Negatives (FN):** Churned customers missed by the model (Type II error)

**Color Coding:**
- Darker blue indicates higher counts
- The diagonal (TN, TP) should have the darkest colors for a good model

**Business Insight:**
- False Negatives are costly: missed churners leave without intervention
- False Positives waste marketing resources on customers who weren't at risk
- Our model minimizes both, with particularly low False Negatives ensuring we catch most at-risk customers

---

### 4. Feature Importance (`feature_importance.png`)

A horizontal bar chart showing the top 20 most influential features in the XGBoost model:

**What it Measures:**
- Feature importance is calculated based on how much each feature contributes to reducing prediction errors
- Higher values indicate features that have more impact on churn predictions
- XGBoost uses "gain" - the average improvement in accuracy brought by a feature when it's used in decision trees

**Top Features Explained:**
1. **LoyaltyLevel_Établi (Established):** Long-term customers are less likely to churn
2. **FavoriteSeason_Hiver (Winter):** Seasonal purchasing patterns strongly predict retention
3. **PreferredMonth:** The month customers typically purchase indicates engagement level
4. **LoyaltyLevel_Nouveau (New):** New customers have different churn patterns
5. **CustomerTenureDays:** How long someone has been a customer

**Business Insight:** This chart guides where to focus retention efforts:
- Loyalty programs should emphasize reaching "Established" status
- Winter marketing campaigns are crucial for retention
- New customer onboarding in their first months is critical

---

### 5. Segment Distribution (`segment_distribution.png`)

A two-panel visualization showing customer segment composition:

**Left Panel - Pie Chart (Customer Distribution):**
- Shows the percentage of customers in each segment identified by KMeans clustering
- Each slice represents one segment with its name and percentage
- Segment names are derived from RFM analysis (Champions, At Risk, etc.)

**Right Panel - Bar Chart (Churn Rate by Segment):**
- X-axis: Customer segments (Cluster 0, 1, 2)
- Y-axis: Churn rate percentage within each segment
- Each bar shows what percentage of customers in that segment have churned
- Percentage labels appear above each bar

**Key Observations:**
- Cluster 0 (At Risk/Churning): 25.4% of customers with 100% churn rate
- Cluster 1 (Champions): 0.6% of customers with 0% churn rate
- Cluster 2 (Active Customers): 74% of customers with 11% churn rate

**Business Insight:** This visualization helps prioritize marketing efforts:
- The large "At Risk" segment (25%) represents immediate intervention opportunities
- Champions, though small, should receive VIP treatment to maintain loyalty
- Active customers need engagement to prevent them from becoming at-risk

---

### 6. Segment Heatmap (`segment_heatmap.png`)

A color-coded matrix showing the normalized RFM profile of each customer segment:

**Structure:**
- Rows: RFM metrics (Recency, Frequency, Monetary)
- Columns: Customer segments with their names
- Cell values: Normalized scores (0.00 to 1.00)

**Color Interpretation:**
- Uses Red-Yellow-Green colormap (reversed)
- For Recency: Lower (greener) is better (more recent = active customer)
- For Frequency: Higher (redder based on inverse) is better (more purchases)
- For Monetary: Higher is better (more spending)

**How to Read:**
- Each cell shows how a segment compares to others on that metric
- 0.00 = lowest among all segments
- 1.00 = highest among all segments
- Annotations show exact normalized values

**Segment Profiles:**
- **Champions:** Low recency (recent), very high frequency and monetary
- **At Risk:** High recency (inactive for long), low frequency and monetary
- **Active Customers:** Moderate across all metrics

**Business Insight:** This heatmap provides a quick diagnostic tool for understanding what makes each segment different, enabling targeted marketing strategies:
- Champions need exclusive rewards matching their high spending
- At Risk customers need re-activation campaigns
- Active customers need nudges to increase frequency

## Authors

Academic project for Machine Learning Retail course.

## License

For educational purposes only.
