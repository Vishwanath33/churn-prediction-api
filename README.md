# Customer Churn Prediction API

End-to-end machine learning pipeline for predicting customer churn with production-ready REST API.

## 🎯 Features
- Trained classification models (Random Forest, XGBoost)
- FastAPI REST endpoint for real-time predictions
- Model evaluation with ROC-AUC, precision, recall
- SMOTE oversampling for imbalanced datasets

## 🛠️ Tech Stack
- Python 3.10+
- Scikit-learn, XGBoost
- FastAPI, Uvicorn
- Pandas, NumPy
- Imbalanced-learn (SMOTE)

## 📊 Model Performance
*(Baseline results on Telco Churn dataset)*
- **Accuracy:** ~82%
- **ROC-AUC:** ~0.85
- **Precision:** ~0.78
- **Recall:** ~0.81

## 🚧 Status
**In Active Development** - API endpoint finalization in progress

### Completed
- [x] Data preprocessing pipeline
- [x] Model training and evaluation
- [x] Baseline model performance

### In Progress
- [ ] FastAPI endpoint implementation
- [ ] API documentation (Swagger UI)
- [ ] Docker containerization
- [ ] Deployment strategy

## 📁 Project Structure
