from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts telecom customer churn using Random Forest",
    version="1.0.0"
)

with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerData):
    df = pd.DataFrame([customer.dict()])

    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(df_aligned)[0]
    probability = model.predict_proba(df_aligned)[0][1]

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }
