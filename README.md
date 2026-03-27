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

### 1. Environment Setup
Create a dedicated virtual environment for this project to avoid dependency conflicts:
```bash
# Create virtual environment
python -m venv cra_env

# Activate environment
# Windows (PowerShell):
.\cra_env\Scripts\Activate.ps1
# Windows (Command Prompt):
cra_env\Scripts\activate.bat
# Mac/Linux:
source cra_env/bin/activate
```

### 2. Install Dependencies
Once the environment is active, install the required packages:
```bash
pip install -r backend/requirements.txt
```

### 3. Train the Machine Learning Model
Generate the inference pipeline and retrain the LightGBM model:
```bash
python backend/train_model.py
```
*(This will save the model to `backend/model_pipeline.joblib`)*

### 4. Start the Web Application
Run the backend server to launch the frontend UI and the prediction API:
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to use the interactive Credit Risk Analyzer!

### 5. Running the Jupyter Notebook Analysis
If you wish to explore the original exploratory data analysis and model comparison:
1. Ensure your virtual environment is activated and dependencies are installed.
2. Open `notebooks/notebook.ipynb` in Jupyter Notebook or VS Code.
3. Run all cells to view the statistical comparisons, EDA visualizations, and validation results.

### Risk Factors
- **Income Level**: Strong inverse correlation with default risk
- **Home Ownership**: Significant differences in default rates across ownership types
- **Loan Characteristics**: Amount, grade and intent influence default probability
- **Demographics**: Age groups show varying risk profiles
