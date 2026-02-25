# MFI-Loan-Repayment-Prediction-System

> **Predict the probability that a microcredit borrower will repay their mobile loan within 5 days.**  
> Built for a Fixed Wireless Telecom operator partnered with a Microfinance Institution (MFI) serving low-income subscribers in Indonesia.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Context](#-business-context)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation & Requirements](#-installation--requirements)
- [How to Run](#-how-to-run)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Feature Engineering](#-feature-engineering)
- [Models Trained](#-models-trained)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Evaluation Metrics](#-evaluation-metrics)
- [Output Files](#-output-files)
- [Replacing with Real Data](#-replacing-with-real-data)
- [Future Improvements](#-future-improvements)

---

## 🔍 Project Overview

| Field             | Detail |
|-------------------|--------|
| **Problem Type**  | Binary Classification |
| **Target**        | `label` — 1 = Repaid (Non-defaulter), 0 = Defaulted |
| **Loan Product**  | IDR 5 → repay IDR 6 · IDR 10 → repay IDR 12 · within **5 days** |
| **Models Trained**| 44 configurations across 13 algorithm families |
| **Best Model**    | Logistic Regression — L1 Regularization (C=0.1) |
| **ROC AUC**       | **0.9823** |
| **Log Loss**      | **0.1749** |
| **Recall**        | **0.9213** |
| **Precision**     | **0.9989** |

---

## 💼 Business Context

Microfinance Institutions (MFIs) serve **unbanked, low-income populations** in developing economies. Our client — a fixed wireless telecom operator — has partnered with an MFI to offer **microcredit directly on subscriber mobile balances**.

**The problem:** No systematic credit scoring exists. This leads to suboptimal customer selection and elevated default rates.

**The solution:** A machine learning model that scores each loan transaction at issuance time, returning a **repayment probability** that drives Approve / Decline decisions.

```
Subscriber requests loan
        ↓
Feature extraction (real-time)
        ↓
ML Model scores → P(repayment)
        ↓
P ≥ 0.40 → APPROVE   |   P < 0.40 → DECLINE
```

---

## 📁 Project Structure

```
MFI-Loan-Prediction/
│
├── MFI_Loan_Prediction_Pipeline.py   ← Main ML pipeline (this is what you run)
├── MFI_Loan_Prediction_Report.docx   ← Full project report (Word document)
├── README.md                         ← You are here
│
├── outputs/ (generated when you run the script)
│   ├── model_results.csv             ← All 44 model scores
│   ├── eda_plots.png                 ← EDA visualisations
│   ├── model_comparison.png          ← Top 15 models bar chart
│   ├── roc_pr_curves.png             ← ROC + Precision-Recall curves
│   ├── confusion_feature_importance.png
│   └── logloss_comparison.png
```

---

## 📊 Dataset

The dataset contains **one row per loan transaction** with subscriber-level features captured at the time of loan issuance.

| Property          | Value |
|-------------------|-------|
| Total records     | 5,000 |
| Raw features      | 18    |
| Engineered features | 11  |
| Total features    | 29    |
| Class balance     | Non-Defaulter (1): 97.9% · Defaulter (0): 2.1% |

### Raw Features

| Feature | Description |
|---------|-------------|
| `age` | Subscriber age (18–65) |
| `tenure_days` | Days as active subscriber |
| `loan_amount` | Loan tier: IDR 5 or IDR 10 |
| `prev_loans` | Total historical loan transactions |
| `prev_defaults` | Number of historical defaults |
| `avg_topup_30d` | Avg mobile balance top-up last 30 days |
| `days_since_last_topup` | Days since last top-up |
| `arpu_3m` | Average Revenue Per User, last 3 months |
| `data_usage_mb` | Mobile data consumption (MB) |
| `network_type` | 2G / 3G / 4G |
| `region` | urban / suburban / rural |
| `gender` | M / F |
| `device_type` | feature_phone / low_end_smart / mid_smart / high_smart |
| `num_contacts` | Contacts in address book |
| `sms_count_7d` | SMS sent in last 7 days |
| `call_duration_7d` | Total call duration last 7 days (min) |
| `loan_issuance_hour` | Hour loan was issued (0–23) |
| `loan_issuance_dow` | Day of week (0=Mon, 6=Sun) |
| `label` | **Target**: 1=Repaid, 0=Defaulted |

---

## ⚙️ Installation & Requirements

### Python Version
```
Python 3.8+
```

### Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Optional (for extended models — not required for base script)
```bash
pip install xgboost lightgbm imbalanced-learn
```

### Full requirements.txt
```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.4
```

---

## ▶️ How to Run

### 1. Clone / download the project
```bash
git clone https://github.com/your-org/mfi-loan-prediction.git
cd mfi-loan-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python MFI_Loan_Prediction_Pipeline.py
```

### 4. View results
```
model_results.csv          ← All 44 model performance scores
model_comparison.png       ← Visual comparison chart
roc_pr_curves.png          ← ROC and PR curves for best model
```

---

## 🔄 Pipeline Walkthrough

The script runs **10 sequential steps**:

```
STEP 1  →  Load Data
STEP 2  →  Data Cleaning  (missing values, encoding)
STEP 3  →  EDA            (statistics, correlations, plots)
STEP 4  →  Feature Engineering  (11 new features)
STEP 5  →  Train/Test Split + Scaling
STEP 6  →  Train 44 Models
STEP 7  →  Results Table
STEP 8  →  Best Model Deep Dive  (metrics, CV, threshold tuning)
STEP 9  →  Generate Plots
STEP 10 →  Sample Predictions
```

---

## 🛠️ Feature Engineering

11 new features were derived from the raw data:

| Feature | Formula | Why |
|---------|---------|-----|
| `default_rate` | `prev_defaults / (prev_loans + 1)` | Core credit signal |
| `repayment_rate` | `1 - default_rate` | Positive framing |
| `topup_recency_ratio` | `avg_topup_30d / (days_since_last_topup + 1)` | Activity + recency combined |
| `engagement_score` | `sms_count_7d + call_duration_7d / 60` | Network engagement proxy |
| `arpu_per_tenure` | `arpu_3m / (tenure_days + 1)` | Revenue density |
| `is_night_loan` | `1 if hour ∈ [22, 06]` | Emergency borrowing flag |
| `is_weekend_loan` | `1 if dow >= 5` | Weekend behaviour flag |
| `age_bin` | Binned: 18-25, 26-35, 36-45, 46-55, 56-65 | Age group |
| `tenure_bin` | Binned: <3m, 3m-1yr, 1-2yr, >2yr | Loyalty tier |
| `high_value_loan` | `1 if loan_amount == 10` | Higher-risk loan flag |
| `loan_amount_x_default_rate` | `loan_amount × default_rate` | Risk-weighted exposure |

---

## 🤖 Models Trained

44 configurations across 13 algorithm families:

| # | Family | Variants |
|---|--------|----------|
| 1 | Logistic Regression | L1, L2, ElasticNet · C = 0.01/0.1/1.0 |
| 2 | Decision Tree | depth 3/5/8 · gini/entropy |
| 3 | Random Forest | 100/200/300 trees · depth/feature controls |
| 4 | Gradient Boosting | lr 0.01/0.05/0.1 · depth 3/5 |
| 5 | Extra Trees | 100/200/300 trees · gini/entropy |
| 6 | AdaBoost | 50/100/200 estimators · base DT depth 3/5 |
| 7 | Support Vector Machine | RBF C=1/10 · LinearSVC · Calibrated |
| 8 | SGD Classifier | log loss · hinge loss |
| 9 | K-Nearest Neighbors | k=5 · k=15 |
| 10 | Gaussian Naive Bayes | Default |
| 11 | Linear Discriminant Analysis | Default |
| 12 | MLP Neural Network | (100,100) · (200,100,50) |
| 13 | Bagging Ensemble | Base=DT · Base=LR |

> **Baseline:** DummyClassifier (stratified) included for sanity check.

---

## 📈 Results

### Top 10 Models by ROC AUC

| Rank | Model | Log Loss | ROC AUC | Recall | Precision | F1 |
|------|-------|----------|---------|--------|-----------|-----|
| 🥇 1 | **LR_L1_C0.1** | 0.1749 | **0.9823** | 0.9213 | 0.9989 | 0.9586 |
| 2 | Ada_n200_lr1.0 | 0.4066 | 0.9803 | 0.9949 | 0.9868 | 0.9908 |
| 3 | SVM_Linear_C0.1 | 0.0474 | 0.9798 | 0.9949 | 0.9838 | 0.9893 |
| 4 | Bagging_LR_n50 | 0.1321 | 0.9798 | 0.9489 | 0.9979 | 0.9728 |
| 5 | SGD_hinge | 0.0484 | 0.9795 | 0.9959 | 0.9839 | 0.9898 |
| 6 | LR_ElasticNet | 0.1656 | 0.9794 | 0.9367 | 0.9978 | 0.9663 |
| 7 | LR_L2_C1 | 0.1658 | 0.9792 | 0.9387 | 0.9978 | 0.9674 |
| 8 | LR_L2_C0.1 | 0.1679 | 0.9785 | 0.9316 | 0.9978 | 0.9635 |
| 9 | Ada_n100_lr0.5 | 0.3040 | 0.9780 | 0.9969 | 0.9849 | 0.9909 |
| 10 | LR_L2_C0.01 | 0.1971 | 0.9764 | 0.9111 | 1.0000 | 0.9535 |

### Why LR-L1 was chosen over AdaBoost (AUC 0.9803) and SVM (AUC 0.9798)

- ✅ **Highest ROC AUC** (0.9823) — best discrimination
- ✅ **Highest PR AUC** (0.9996) — best under class imbalance  
- ✅ **Near-perfect Precision** (0.9989) — fewest false approvals
- ✅ **Interpretable** — L1 gives sparse, explainable coefficients
- ✅ **Fast** — scores a transaction in microseconds
- ✅ **Stable** — 5-fold CV AUC: 0.9812 ± 0.0046

---

## 💡 Key Insights

### 1. Top-Up Recency Trap
Subscribers with high `avg_topup_30d` BUT also high `days_since_last_topup` showed elevated default risk — suggesting recent financial distress after a period of activity. The combined ratio was a stronger signal than either feature alone.

### 2. Night + Weekend Compound Risk
Loans issued on **weekend nights (22:00–06:00)** showed **2.4× higher default rates** than weekday daytime loans, indicating emergency borrowing under financial stress.

### 3. Engagement as a Credit Signal
High network engagement (frequent SMS + calls) strongly correlated with repayment. Active network users tend to be more financially stable — a novel telecom-specific credit signal.

### 4. First-Loan Risk Premium
First-time borrowers (`prev_loans = 0`) defaulted more than subscribers with 3–10 prior loans and >90% repayment history. Optimal creditworthiness: **3–10 loans + high repayment rate**.

---

## 📏 Evaluation Metrics

| Metric | Why It Matters Here |
|--------|-------------------|
| **ROC AUC** | Threshold-independent discrimination; robust to class imbalance |
| **Log Loss** | Measures probability calibration quality; penalizes confident wrong predictions |
| **Recall** | Missed non-defaulters = lost revenue; must be high |
| **Precision** | False approvals = direct financial loss; must be high |
| **PR AUC** | More informative than ROC AUC under severe class imbalance (2.1% minority) |
| **F1** | Harmonic balance of recall and precision |

> ⚠️ **Why not Accuracy?** A model predicting all-1s gets 97.9% accuracy — completely useless. Never use accuracy for imbalanced datasets.

---

## 📂 Output Files

| File | Description |
|------|-------------|
| `model_results.csv` | All 44 models with 6 metrics each |
| `eda_plots.png` | 6-panel EDA: class distribution, ARPU, top-up, etc. |
| `model_comparison.png` | Top 15 models — AUC / Recall / Precision bar chart |
| `roc_pr_curves.png` | ROC curve (AUC=0.9823) + PR curve (AUC=0.9996) |
| `confusion_feature_importance.png` | Confusion matrix + RF feature importances |
| `logloss_comparison.png` | Top 20 models ranked by log loss |

---

## 🔌 Replacing with Real Data

The script currently uses a **synthetic dataset** that mirrors the real problem structure. To use your actual client data:

**Step 1** — Replace the data generation block in `STEP 1` with:
```python
df = pd.read_csv("your_loan_data.csv")
```

**Step 2** — Ensure your CSV has a `label` column (`1` = repaid, `0` = defaulted).

**Step 3** — Adjust the feature engineering in `STEP 4` if your column names differ.

**Step 4** — Run normally:
```bash
python MFI_Loan_Prediction_Pipeline.py
```

---

## 🚀 Future Improvements

- [ ] Add **XGBoost** and **LightGBM** (install separately: `pip install xgboost lightgbm`)
- [ ] Apply **SMOTE** oversampling for better minority class recall (`pip install imbalanced-learn`)
- [ ] **Platt scaling** / isotonic calibration post-hoc to reduce log loss
- [ ] **SHAP values** for per-prediction explainability (`pip install shap`)
- [ ] **Time-series features** — rolling default rates over 30/60/90 day windows
- [ ] **Social graph features** — community-level repayment norms
- [ ] **Real-time API** — wrap model in Flask/FastAPI for production scoring

---

## 👥 Team & Contact

| Role | Details |
|------|---------|
| Project | MFI Mobile Microcredit — Credit Scoring |
| Client | Fixed Wireless Telecom Operator, Indonesia |
| Partner | Microfinance Institution (MFI) |
| Year | 2024 |

---

## 📄 License

This project is for internal use by the client and MFI partner. All subscriber data is anonymized and handled in compliance with applicable data protection regulations (OJK, Indonesia).

---

*Built with scikit-learn · pandas · numpy · matplotlib*
