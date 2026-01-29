import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("🏦 Customer Churn Prediction")

# Load model function
@st.cache_resource
def load_model():
    try:
        import joblib
        model = joblib.load('new_model.pkl')
        return model
    except:
        try:
            import pickle
            with open('new_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except:
            return "no model found"

# Load scaler function
@st.cache_resource
def load_scaler():
    try:
        import joblib
        scaler = joblib.load('scaler.pkl')
        return scaler
    except:
        try:
            import pickle
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return scaler
        except:
            # Create simple scaler if not found
            scaler = StandardScaler()
            dummy_data = np.random.randn(100, 15)
            scaler.fit(dummy_data)
            return scaler

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Input form
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=619)
    Age = st.number_input("Age", min_value=18, max_value=100, value=42)
    Tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=2)
    Balance = st.number_input("Balance", min_value=0.0, value=0.0, format="%.2f")
    EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=101348.88, format="%.2f")

with col2:
    CreditUtilization = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.0, format="%.3f")
    InteractionScore = st.number_input("Interaction Score", min_value=0, max_value=10, value=3)
    HasCrCard = st.selectbox("Has Credit Card", ["Yes", "No"], index=0)
    NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    IsActiveMember = st.selectbox("Is Active Member", ["Yes", "No"], index=0)

# Additional inputs
Gender = st.selectbox("Gender", ["Female", "Male"], index=0)
Geography = st.radio("Geography", ["France", "Germany", "Spain"], index=0)

# Calculate derived feature
BalanceToSalary = Balance / EstimatedSalary if EstimatedSalary > 0 else 0

# Convert categorical to numerical
HasCrCard_num = 1 if HasCrCard == "Yes" else 0
IsActiveMember_num = 1 if IsActiveMember == "Yes" else 0
Gender_num = 1 if Gender == "Male" else 0

# Geography one-hot encoding
Geography_France = 1 if Geography == "France" else 0
Geography_Germany = 1 if Geography == "Germany" else 0
Geography_Spain = 1 if Geography == "Spain" else 0

# Prepare input data
input_data = {
    'CreditScore': float(CreditScore),
    'Gender': float(Gender_num),
    'Age': float(Age),
    'Tenure': float(Tenure),
    'Balance': float(Balance),
    'NumOfProducts': float(NumOfProducts),
    'HasCrCard': float(HasCrCard_num),
    'IsActiveMember': float(IsActiveMember_num),
    'EstimatedSalary': float(EstimatedSalary),
    'CreditUtilization': float(CreditUtilization),
    'InteractionScore': float(InteractionScore),
    'BalanceToSalary': float(BalanceToSalary),
    'Geography_France': float(Geography_France),
    'Geography_Germany': float(Geography_Germany),
    'Geography_Spain': float(Geography_Spain)
}

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Ensure correct column order
expected_features = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
    'EstimatedSalary', 'CreditUtilization', 'InteractionScore',
    'BalanceToSalary', 'Geography_France', 'Geography_Germany', 
    'Geography_Spain'
]

input_df = input_df[expected_features]

# Predict button
st.markdown("---")
if st.button("Predict Churn", type="primary", use_container_width=True):
    try:
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Display result
        if prediction == 1:
            st.error(f"## 🔴 Customer WILL CHURN")
        else:
            st.success(f"## 🟢 Customer WILL NOT CHURN")
        
        # Show probability
        churn_prob = prediction_proba[1] * 100
        st.info(f"**Churn Probability:** {churn_prob:.2f}%")
        
        # Show confidence
        confidence = max(prediction_proba) * 100
        st.info(f"**Confidence:** {confidence:.2f}%")
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Footer
st.markdown("---")
st.caption("Customer Churn Prediction Model")