import os
import streamlit as st
import requests
import json
import pandas as pd

# Page Config
st.set_page_config(page_title="Sentinel Churn Predictor", layout="centered")

# Title
st.title("🛡️ Project Sentinel: Customer Retention Engine")
st.markdown("Real-time scoring of customer churn risk using XGBoost & FastAPI.")

# Sidebar for Inputs
st.sidebar.header("Customer Profile")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 90, 30)
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 20.0, 200.0, 70.0)
    usage = st.sidebar.number_input("Avg Daily Usage (Mins)", 0, 500, 50)
    payment_fails = st.sidebar.selectbox("Payment Failures (Last 3 Mo)", [0, 1, 2, 3])
    sentiment = st.sidebar.slider("Last Interaction Sentiment", -1.0, 1.0, 0.0)
    
    data = {
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "avg_daily_usage_min": usage,
        "payment_fails_last_3m": payment_fails,
        "last_interaction_sentiment": sentiment
    }
    return data

input_data = user_input_features()

# Display Input Data
st.subheader("Customer Data")
st.write(pd.DataFrame([input_data]))
# Button to Predict
if st.button("Assess Churn Risk"):
    # GET THE URL FROM ENVIRONMENT VARIABLE (OR DEFAULT TO LOCALHOST)
    # This allows us to configure the URL in Render's settings later
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
    endpoint = f"{api_url}/predict_churn"

    try:
        response = requests.post(endpoint, json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            probability = result['churn_probability']
            label = result['risk_label']
            explanation = result['explanation'] # <--- Get SHAP data
            
            # --- 1. Main Result ---
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Churn Probability", value=f"{probability:.2%}")
            with col2:
                if label == "High Risk":
                    st.error(f"⚠️ {label}")
                else:
                    st.success(f"✅ {label}")
            
            st.progress(probability)
            
            # --- 2. Explainability (New Section) ---
            st.subheader("🤖 Why did the AI make this decision?")
            
            # CHECK: Do we actually have an explanation?
            if explanation and "error" not in explanation:
                st.caption("Positive values (Red) increase risk. Negative values (Blue) decrease risk.")
                
                # Convert dictionary to DataFrame for plotting
                shap_df = pd.DataFrame(list(explanation.items()), columns=['Feature', 'Impact'])
                
                # Sort by absolute impact
                shap_df['Abs_Impact'] = shap_df['Impact'].abs()
                shap_df = shap_df.sort_values('Abs_Impact', ascending=False).drop(columns=['Abs_Impact'])
                
                # Custom Bar Chart
                st.bar_chart(shap_df.set_index('Feature'))
                
                # Narrative Explanation (Safe Check)
                if not shap_df.empty:
                    top_factor = shap_df.iloc[0]
                    action = "increasing" if top_factor['Impact'] > 0 else "decreasing"
                    st.markdown(f"**Key Insight:** The main factor is **{top_factor['Feature']}**, which is **{action}** the risk.")
            else:
                # Fallback if SHAP is disabled
                st.info("ℹ️ Detailed AI explanation is currently unavailable (SHAP engine is offline).")
            
    except Exception as e:
        st.error(f"Connection Error: {e}")