# app.py
import streamlit as st
import pandas as pd
import joblib
import logging
from datetime import datetime
import os
import sys
import numpy as np

# --- 1. Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("churn_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- 2. Paths ---
MODEL_PATH = os.path.join("models", "churn_model.pkl")

# --- 3. Preprocessing (must match training exactly) ---
def preprocess_input(input_dict):
    """Preprocess user input to match model training format"""
    df = pd.DataFrame([input_dict])
    
    # Feature engineering (must match training)
    df['Is_Young_Adult'] = (df['Age'] < 30).astype(int)
    df['Is_Senior'] = (df['Age'] > 60).astype(int)
    df['Is_New_Customer'] = (df['Tenure'] < 6).astype(int)
    df['Is_Mid_Tenure'] = df['Tenure'].between(6, 24).astype(int)
    df['High_Support_Calls'] = (df['Support_Calls'] > 5).astype(int)
    df['Medium_Support_Calls'] = df['Support_Calls'].between(2, 5).astype(int)
    df['Support_Calls_per_Usage'] = df['Support_Calls'] / (df['Usage_Frequency'] + 1)
    df['Has_Payment_Delay_Issue'] = (df['Payment_Delay'] > 3).astype(int)
    df['Is_Female'] = (df['Gender'] == 'Female').astype(int)
    df['Female_High_Support'] = ((df['Gender'] == 'Female') & (df['Support_Calls'] > 5)).astype(int)
    df['Is_Monthly_Contract'] = (df['Contract_Length'] == 'Monthly').astype(int)
    df['Is_High_Spender'] = (df['Total_Spend'] > 700).astype(int)

    # Categorical binning
    df['AgeGroup'] = pd.cut(df['Age'], 
                          bins=[0, 25, 45, 65, 100], 
                          labels=['18-25', '26-45', '46-65', '65+'])
    df['TenureGroup'] = pd.cut(df['Tenure'], 
                             bins=[0, 6, 24, 60], 
                             labels=['0-6', '7-24', '25+'])
    df['Payment_Delay_Group'] = pd.cut(df['Payment_Delay'], 
                                    bins=[-1, 1, 3, 10], 
                                    labels=['Low', 'Medium', 'High'])

    # One-hot encoding
    categorical_cols = {
        'Gender': ['Male', 'Female'],
        'Subscription_Type': ['Basic', 'Standard', 'Premium'],
        'Contract_Length': ['Monthly', 'Quarterly', 'Annual'],
        'AgeGroup': ['18-25', '26-45', '46-65', '65+'],
        'TenureGroup': ['0-6', '7-24', '25+'],
        'Payment_Delay_Group': ['Low', 'Medium', 'High']
    }

    for col, categories in categorical_cols.items():
        for category in categories:
            df[f"{col}_{category}"] = (df[col] == category).astype(int)
    
    return df

# --- 4. Model Loading ---
@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Test prediction to verify model works
        test_input = {
            'Age': 35,
            'Tenure': 12,
            'Usage_Frequency': 15,
            'Support_Calls': 2,
            'Payment_Delay': 1,
            'Total_Spend': 800,
            'Last_Interaction': 7,
            'Gender': 'Male',
            'Subscription_Type': 'Standard',
            'Contract_Length': 'Annual'
        }
        test_df = preprocess_input(test_input)
        _ = model.predict_proba(test_df)
        
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        st.error("Failed to load prediction model")
        st.stop()

# --- 5. Prediction Function ---
def predict_churn(model, input_data):
    try:
        # Preprocess input
        processed_data = preprocess_input(input_data)
        
        # Ensure all expected features are present
        missing_cols = set(model.feature_names_in_) - set(processed_data.columns)
        for col in missing_cols:
            processed_data[col] = 0  # Add missing columns with default value
        
        # Reorder columns to match training
        processed_data = processed_data[model.feature_names_in_]
        
        # Make prediction
        churn_prob = model.predict_proba(processed_data)[0][1]
        
        return {
            'probability': churn_prob,
            'risk_level': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.4 else 'Low'
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise

# --- 6. GUI Components ---
def user_input_form():
    st.sidebar.header("Customer Details")
    return {
        'Age': st.sidebar.slider("Age", 18, 80, 35),
        'Tenure': st.sidebar.slider("Tenure (months)", 0, 60, 12),
        'Usage_Frequency': st.sidebar.slider("Usage Frequency", 0, 100, 15),
        'Support_Calls': st.sidebar.slider("Support Calls", 0, 20, 2),
        'Payment_Delay': st.sidebar.slider("Payment Delay (days)", 0, 30, 5),
        'Total_Spend': st.sidebar.number_input("Total Spend ($)", 0, 10000, 500),
        'Last_Interaction': st.sidebar.slider("Last Interaction (days ago)", 0, 30, 7),
        'Gender': st.sidebar.selectbox("Gender", ["Male", "Female"]),
        'Subscription_Type': st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"]),
        'Contract_Length': st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    }

def display_results(prediction):
    st.subheader("Prediction Results")
    
    # Probability gauge
    st.metric("Churn Probability", f"{prediction['probability']:.1%}")
    st.progress(int(prediction['probability'] * 100))
    
    # Risk level
    if prediction['risk_level'] == 'High':
        st.error("🚨 High Risk of Churn")
        st.write("**Recommended Actions:** Immediate retention offer, personal outreach, account review")
    elif prediction['risk_level'] == 'Medium':
        st.warning("⚠️ Medium Risk of Churn")
        st.write("**Recommended Actions:** Proactive engagement, special offers, usage optimization")
    else:
        st.success("✅ Low Risk of Churn")
        st.write("**Recommended Actions:** Regular check-ins, value-added services")

# --- 7. Main Application ---
def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
    st.title("Customer Churn Prediction Dashboard")
    
    # Load model first
    model = load_model()
    
    # User input
    user_data = user_input_form()
    
    # Prediction button
    if st.sidebar.button("Predict Churn Risk", type="primary"):
        with st.spinner("Analyzing customer data..."):
            try:
                prediction = predict_churn(model, user_data)
                display_results(prediction)
                
                # Show test cases
                st.subheader("Example Test Cases")
                cols = st.columns(3)
                
                # Low risk example
                with cols[0]:
                    st.write("**Low Risk Case**")
                    st.code("""Age: 35
Tenure: 24
Usage: 20
Support Calls: 1
Payment Delay: 1
Spend: 800
Last Contact: 7
Gender: Male
Subscription: Premium
Contract: Annual""")
                
                # Medium risk example
                with cols[1]:
                    st.write("**Medium Risk Case**")
                    st.code("""Age: 45
Tenure: 6
Usage: 10
Support Calls: 4
Payment Delay: 5
Spend: 500
Last Contact: 14
Gender: Female
Subscription: Standard
Contract: Quarterly""")
                
                # High risk example
                with cols[2]:
                    st.write("**High Risk Case**")
                    st.code("""Age: 65
Tenure: 3
Usage: 5
Support Calls: 8
Payment Delay: 10
Spend: 300
Last Contact: 28
Gender: Female
Subscription: Basic
Contract: Monthly""")
                
            except Exception as e:
                st.error("Prediction failed. Please check the input values.")

if __name__ == "__main__":
    main()
