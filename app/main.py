from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import numpy as np
import shap

app = FastAPI(title="Loan Default Predictor API")


pipeline = joblib.load("models/loan_default_pipeline.joblib")
feature_names = joblib.load("models/feature_names.joblib")
background_data = joblib.load("models/shap_background.joblib")

classifier = pipeline.named_steps["classifier"]
preprocessed_background = pipeline[:-1].transform(background_data)
explainer = shap.LinearExplainer(classifier, preprocessed_background)

PURPOSE_OPTIONS = [
    "all_other", "credit_card", "debt_consolidation",
    "educational", "home_improvement", "major_purchase", "small_business",
]


class LoanApplication(BaseModel):
    annual_income: float = Field(..., gt=0, description="Annual income in dollars")
    dti: float = Field(..., ge=0, description="Debt-to-income ratio, e.g. 15.3 means 15.3%")
    fico: int = Field(..., ge=300, le=850, description="FICO credit score")
    revol_bal: float = Field(..., ge=0, description="Revolving balance in dollars")
    revol_util: float = Field(..., ge=0, description="Revolving utilization percent")
    inq_last_6mths: int = Field(..., ge=0, description="Credit inquiries in the last 6 months")
    delinq_2yrs: int = Field(..., ge=0, description="Delinquencies in the past 2 years")
    pub_rec: int = Field(..., ge=0, description="Derogatory public records")
    years_with_cr_line: float = Field(..., ge=0, description="Years of credit history")
    purpose: Literal[
        "all_other", "credit_card", "debt_consolidation",
        "educational", "home_improvement", "major_purchase", "small_business",
    ]

def build_model_input(app_data: LoanApplication) -> pd.DataFrame:
    """Convert raw applicant facts into the exact engineered feature set the model expects."""
    log_annual_inc = np.log(app_data.annual_income)
    revol_bal_to_income = app_data.revol_bal / app_data.annual_income

    row = {
        "log.annual.inc": log_annual_inc,
        "dti": app_data.dti,
        "fico": app_data.fico,
        "revol.bal": app_data.revol_bal,
        "revol.util": app_data.revol_util,
        "inq.last.6mths": app_data.inq_last_6mths,
        "delinq.2yrs": app_data.delinq_2yrs,
        "pub.rec": app_data.pub_rec,
        "revol.bal_to_income": revol_bal_to_income,
        "years_with_cr_line": app_data.years_with_cr_line,
    }

    for option in PURPOSE_OPTIONS:
        if option == "all_other":
            continue  
        row[f"purpose_{option}"] = 1 if app_data.purpose == option else 0

    return pd.DataFrame([row])[feature_names]

@app.get("/")
def root():
    return {"message": "Loan Default Predictor API is running"}

@app.post("/predict")
def predict(application: LoanApplication):
    input_df = build_model_input(application)
    probability = pipeline.predict_proba(input_df)[0, 1]

   
    preprocessed_input = pipeline[:-1].transform(input_df)
    shap_result = explainer(preprocessed_input)

  
    explanation = [
        {
            "feature": name,
            "value": round(float(val), 4),
            "contribution": round(float(shap_val), 4),
        }
        for name, val, shap_val in zip(
            feature_names, input_df.iloc[0], shap_result.values[0]
        )
    ]
    explanation.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    return {
        "default_probability": round(float(probability), 4),
        "risk_tier": "High Risk" if probability >= 0.5 else "Low Risk",
        "explanation": explanation,
    }