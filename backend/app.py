from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Credit Risk Analyzer API", description="API to predict credit risk based on customer financial data.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pipeline.joblib")
try:
    pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
os.makedirs(frontend_dir, exist_ok=True)

class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int
    
    model_config = ConfigDict(extra='ignore')

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(frontend_dir, "index.html")
    if not os.path.exists(index_path):
        return {"message": "Frontend not built yet. Create frontend/index.html"}
    return FileResponse(index_path)

@app.post("/predict")
def predict_risk(application: LoanApplication):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded")
    
    df = pd.DataFrame([application.model_dump()])
    
    # Feature Generation
    df['age_group'] = pd.cut(df['person_age'], bins=[0, 26, 36, 46, 56, 66, 120], labels=['20-25', '26-35', '36-45', '46-55', '56-65', '66+']).astype(str)
    df['income_group'] = pd.cut(df['person_income'], bins=[0, 25000, 50000, 75000, 100000, float('inf')], labels=['low', 'low-middle', 'middle', 'high-middle', 'high']).astype(str)
    df['loan_amount_group'] = pd.cut(df['loan_amnt'], bins=[0, 5000, 10000, 15000, float('inf')], labels=['small', ' medium', 'large', 'very large']).astype(str)
    
    df['loan_to_income_ratio'] = np.where(df['person_income'] > 0, df['loan_amnt'] / df['person_income'], 0)
    df['loan_to_emp_length_ratio'] = np.where(df['loan_amnt'] > 0, df['person_emp_length'] / df['loan_amnt'], 0)
    df['int_rate_to_loan_amt_ratio'] = np.where(df['loan_amnt'] > 0, df['loan_int_rate'] / df['loan_amnt'], 0)
    
    # Ensure all missing are filled if any (shouldn't be, but just in case)
    df.fillna(0, inplace=True)
    
    # Add dummy loan percent income since notebook had it (loan_amnt / person_income)
    # Actually the notebook also had loan_percent_income already in the dataset.
    df['loan_percent_income'] = np.where(df['person_income'] > 0, df['loan_amnt'] / df['person_income'], 0)
    
    try:
        prob = pipeline.predict_proba(df)[0][1]
        prediction_class = int(pipeline.predict(df)[0])
        
        if prob <= 0.3:
            risk_category = "Low Risk"
        elif prob <= 0.7:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
            
        return {
            "prediction": prediction_class,
            "probability": float(prob),
            "risk_category": risk_category
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
