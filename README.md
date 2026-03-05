# Credit Risk Analysis - Predictive Modeling Case Study

A comprehensive machine learning project for credit risk assessment using advanced classification techniques and robust model evaluation.

## Project Objective

Develop and evaluate machine learning models to predict loan default risk, enabling financial institutions to make data-driven lending decisions and minimize credit losses.

## 📊 Dataset Overview

- **Source**: Credit risk dataset with borrower and loan characteristics
- **Target Variable**: `loan_status` (0: No Default, 1: Default)
- **Features**: Demographics, financial information, loan details, employment history

### Key Features
- **Personal**: Age, income, home ownership, employment length
- **Loan**: Amount, interest rate, grade, intent
- **Engineered**: Age groups, income brackets, loan-to-income ratios

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of target and feature variables
- Correlation matrix and statistical relationships
- Visualization of risk patterns across different segments

### 2. Feature Engineering
- **Categorical Groupings**: Age ranges, income brackets, loan amount tiers
- **Financial Ratios**: Loan-to-income, interest rate-to-loan amount ratios
- **Statistical Validation**: Chi-square tests for feature significance

### 3. Data Preprocessing
- **Missing Data**: Handled through strategic imputation
- **Outlier Treatment**: Removed unrealistic age values (>80 years)
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for numerical features

### 4. Model Development
Implemented and compared 9 classification algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM

### 5. Model Evaluation
- **Cross-Validation**: 5-fold StratifiedKFold for robust performance estimation
- **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- **Metrics**: Accuracy, Precision, Recall, F1-score with business context
- **Business Focus**: Prioritized Recall (detecting defaults) over Precision

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
1. Clone this repository
2. Install dependencies
3. Open `notebooks/notebook.ipynb` in Jupyter
4. Run all cells for complete analysis

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Risk Factors
- **Income Level**: Strong inverse correlation with default risk
- **Home Ownership**: Significant differences in default rates across ownership types
- **Loan Characteristics**: Amount, grade and intent influence default probability
- **Demographics**: Age groups show varying risk profiles
