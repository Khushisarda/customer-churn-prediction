import streamlit as st
import joblib
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🤖 Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")

# --- Define File Paths ---
# --- Define File Paths ---
SCALER_FILE = "scaler.pkl"
MODEL_FILE = "xgboost_model.pkl"

# --- Helper function to load models safely ---
@st.cache_resource
def load_model(file_path):
    """Loads a .pkl file safely with error handling."""
    if not os.path.exists(file_path):
        st.error(f"Error: Model file not found at '{file_path}'")
        return None
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {file_path}: {e}")
        return None

# --- Load all models ---
scaler = load_model(SCALER_FILE)
model = load_model(MODEL_FILE)

# Stop the app if any model failed to load
if not all([scaler, model]):
    st.warning("One or more model files could not be loaded. The app cannot proceed.")
    st.stop()
else:
    st.success("All models (Scaler & XGBoost) loaded successfully!")


# --- Streamlit UI ----
# These inputs must match the total features your model was trained on
# We assume 11 total features: 5 categorical + 6 numerical

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Demographics")
    
    # --- 5 Categorical Features (Assumed) ---
    # We are guessing these are your 5 categorical features.
    # If they are different, change the st.selectbox labels.
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents_cat = st.selectbox("Has Dependents (Categorical)?", ["No", "Yes"]) # e.g., 'Dependents'
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ]) # e.g., 'PaymentMethod'

with col2:
    st.subheader("📈 Customer Account Data")
    
    # --- 6 Numerical Features (From your training code) ---
    age = st.number_input("Age", min_value=18, value=30)
    num_dependents = st.number_input("Number of Dependents", min_value=0, value=0) # e.g., 'Number of Dependents'
    num_referrals = st.number_input("Number of Referrals", min_value=0, value=0)
    tenure = st.number_input("Tenure in Months", min_value=0, value=0)
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, value=0.00, format="%.2f")
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=0.00, format="%.2f")


# --- Results Area ---
st.divider()
st.subheader("🔍 Prediction")

if st.button("Predict Churn"):
    
    try:
        # --- 1. Process Categorical Features ---
        # Manually encode the 5 categorical features.
        # This order MUST match the order they appeared in your training data.
        gender_map = {'Male': 1, 'Female': 0}
        yes_no_map = {'Yes': 1, 'No': 0}
        payment_map = {
            "Electronic check": 0, 
            "Mailed check": 1,
            "Bank transfer (automatic)": 2, 
            "Credit card (automatic)": 3
        } # Assuming 0, 1, 2, 3 encoding

        cat_features = np.array([[
            gender_map[gender],
            yes_no_map[senior_citizen],
            yes_no_map[partner],
            yes_no_map[dependents_cat],
            payment_map[payment_method]
        ]])

        # --- 2. Process Numerical Features ---
        # Create the 6-feature array for the scaler.
        # This order MUST match your 'cols_to_scale' list.
        num_features_unscaled = np.array([[
            age,
            num_dependents,
            num_referrals,
            tenure,
            monthly_charge,
            total_charges
        ]])
        
        # --- 3. Apply Scaler ---
        # Scale ONLY the 6 numerical features. This fixes the error.
        num_features_scaled = scaler.transform(num_features_unscaled)

        # --- 4. Combine Features ---
        # Combine the 5 categorical features with the 6 scaled numerical features.
        # The order (e.g., categorical first) must match your model's training.
        final_features = np.concatenate([cat_features, num_features_scaled], axis=1)

        # --- 5. Prediction ---
        # The model should now receive the 11 features it expects.
        prediction = model.predict(final_features)
        prediction_proba = model.predict_proba(final_features)
        
        result_class = prediction[0]
        proba_churn = prediction_proba[0][1] 

        # --- 6. Display Results ---
        if result_class == 1:
            st.error(f"**Prediction: Customer will CHURN**")
            st.metric(label="Churn Confidence", value=f"{proba_churn * 100:.2f}%")
        else:
            st.success(f"**Prediction: Customer will STAY**")
            st.metric(label="Stay Confidence", value=f"{(1 - proba_churn) * 100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred during preprocessing or prediction: {e}")
        st.warning("""
            **Error Note:** If you still get an error, it's likely one of these:
            1.  **Feature Order:** The order of the 5 categorical features or 6 numerical features is wrong.
            2.  **Model Features:** Your `xgboost_model.pkl` is *still* the old one expecting 928 features.
            
            **Solution:** Re-run your training notebook *one last time* to save all files, ensuring your model is trained on the 11 features defined here.
        """)

else:
    st.info("Click the 'Predict Churn' button to see the result.")