import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import joblib
import os

def generate_features(X_out):
    X_out['age_group'] = pd.cut(X_out['person_age'], bins=[0, 26, 36, 46, 56, 66, 120], labels=['20-25', '26-35', '36-45', '46-55', '56-65', '66+']).astype(str)
    X_out['income_group'] = pd.cut(X_out['person_income'], bins=[0, 25000, 50000, 75000, 100000, float('inf')], labels=['low', 'low-middle', 'middle', 'high-middle', 'high']).astype(str)
    X_out['loan_amount_group'] = pd.cut(X_out['loan_amnt'], bins=[0, 5000, 10000, 15000, float('inf')], labels=['small', ' medium', 'large', 'very large']).astype(str)
    
    X_out['loan_to_income_ratio'] = np.where(X_out['person_income'] > 0, X_out['loan_amnt'] / X_out['person_income'], 0)
    X_out['loan_to_emp_length_ratio'] = np.where(X_out['loan_amnt'] > 0, X_out['person_emp_length'] / X_out['loan_amnt'], 0)
    X_out['int_rate_to_loan_amt_ratio'] = np.where(X_out['loan_amnt'] > 0, X_out['loan_int_rate'] / X_out['loan_amnt'], 0)
    return X_out

def main():
    print("Loading data...")
    df = pd.read_csv("datasets/credit_risk_dataset.csv")
    
    df.dropna(axis=0, inplace=True)
    df = df[df['person_age'] <= 80]
    
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # Generate features before pipeline
    X = generate_features(X)
    
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file', 'age_group', 'income_group', 'loan_amount_group']
    numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_to_income_ratio', 'loan_to_emp_length_ratio', 'int_rate_to_loan_amt_ratio']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print("Training pipeline...")
    pipeline.fit(X, y)
    
    print(f"Model trained! Training Accuracy: {pipeline.score(X, y):.4f}")
    
    output_path = "backend/model_pipeline.joblib"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
