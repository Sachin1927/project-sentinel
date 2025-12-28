import sys
# Block PySpark on Windows to prevent SHAP crashes
sys.modules["pyspark"] = None 

import pandas as pd
import joblib
import logging
import shap
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# --- CONFIG ---
MODEL_PATH = Path("models/production/xgb_churn_v1.joblib")

# --- 1. ROBUST RESOURCE LOADING ---
model = None
explainer = None

try:
    # Load the Model (Critical - Must Succeed)
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ FATAL: Could not load model. {e}")
    raise RuntimeError("Model loading failed")

try:
    # Load SHAP (Optional - Can Fail safely)
    explainer = shap.TreeExplainer(model)
    logger.info("✅ SHAP Explainer loaded.")
except Exception as e:
    logger.warning(f"⚠️ SHAP initialization failed. Explainability will be disabled. Error: {e}")
    explainer = None  # We continue without SHAP

# --- 2. API DEFINITION ---
class CustomerData(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    avg_daily_usage_min: int
    payment_fails_last_3m: int
    last_interaction_sentiment: float
    # Note: timestamp columns are not needed for inference, only training

app = FastAPI(title="Sentinel Churn Predictor", version="1.2")

@app.post("/predict_churn")
def predict(data: CustomerData):
    try:
        if not model:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Prepare Input
        input_data = data.model_dump()
        input_df = pd.DataFrame([input_data])
        
        # 1. Prediction
        probability = model.predict_proba(input_df)[:, 1][0]
        risk_label = "High Risk" if probability > 0.5 else "Safe"
        
        # 2. Explanation (SHAP) - SAFE MODE
        explanation = {}
        if explainer:
            try:
                shap_values = explainer.shap_values(input_df)
                # Handle different SHAP return types (array vs list)
                if isinstance(shap_values, list):
                    vals = shap_values[0]
                else:
                    vals = shap_values[0] # For binary classification, sometimes it's index 0
                
                explanation = dict(zip(input_df.columns, vals.tolist()))
            except Exception as e:
                logger.warning(f"SHAP calculation failed during request: {e}")
                explanation = {"error": "Explanation unavailable"}

        return {
            "churn_probability": round(float(probability), 4),
            "risk_label": risk_label,
            "explanation": explanation 
        }
    
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))