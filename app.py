import streamlit as st
import joblib
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Enhanced Custom Styling ---
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    
    /* Soft gradient background with texture - Muted blues and purples */
    .stApp {
        background: 
            linear-gradient(135deg, rgba(44, 62, 80, 0.95) 0%, rgba(52, 73, 94, 0.95) 50%, rgba(74, 95, 127, 0.95) 100%),
            repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(0, 0, 0, 0.03) 10px, rgba(0, 0, 0, 0.03) 20px),
            #2c3e50 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .main {
        background: transparent !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Hero section - Softer colors */
    .hero-section {
        background: linear-gradient(135deg, rgba(108, 92, 231, 0.2), rgba(130, 119, 237, 0.15)) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 24px !important;
        padding: 3rem 2rem !important;
        margin-bottom: 2.5rem !important;
        border: 2px solid rgba(139, 92, 246, 0.3) !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
    }

    h1 {
        text-align: center !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 500;
        line-height: 1.6;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
    }

    /* Section headings */
    h3 {
        color: #e0e7ff !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
        font-size: 1.6rem !important;
        text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Card containers - Soft lavender/periwinkle */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        background: linear-gradient(135deg, #c7d2fe 0%, #a5b4fc 100%) !important;
        padding: 2.5rem !important;
        border-radius: 24px !important;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.25) !important;
        border: 2px solid rgba(165, 180, 252, 0.4) !important;
        margin-bottom: 1rem !important;
    }

    /* Card headings */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] h4 {
        color: #1e293b !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.8rem !important;
        border-bottom: 4px solid #4338ca !important;
        text-shadow: none !important;
    }

    /* Fix for spacing */
    div[data-testid="stVerticalBlock"] > div[style*="gap"] {
        gap: 0 !important;
    }
    
    /* Input labels */
    label, [data-testid="stWidgetLabel"] {
        color: #1e293b !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: none !important;
    }
    
    .stSelectbox label, .stNumberInput label {
        display: block !important;
        color: #1e293b !important;
        font-weight: 800 !important;
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput {
        margin-bottom: 1.2rem !important;
    }
    
    input[type="number"] {
        background: #ffffff !important;
        border: 3px solid #4338ca !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-weight: 700 !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
    }
    
    select {
        background: #ffffff !important;
        border: 3px solid #4338ca !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-weight: 700 !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
    }
    
    input:focus, select:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
        outline: none !important;
    }

    /* Dropdown */
    [data-baseweb="select"] {
        background: #ffffff !important;
        border-radius: 12px !important;
    }
    
    [data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 3px solid #4338ca !important;
        border-radius: 12px !important;
    }

    /* Number input buttons */
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        background: #4338ca !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
    }

    /* Hide those extra boxes beside button */
    div[data-testid="column"]:has(.stButton) ~ div[data-testid="column"] {
        display: none !important;
    }
    
    div[data-testid="column"]:has(.stButton) {
        flex-grow: 1 !important;
        max-width: 100% !important;
    }

    /* Main button - Softer coral/salmon gradient */
    .stButton > button {
        background: linear-gradient(135deg, #fb923c 0%, #f472b6 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 1.3rem 4rem !important;
        font-size: 1.25rem !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 15px 40px rgba(251, 146, 60, 0.35) !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 20px 50px rgba(251, 146, 60, 0.5) !important;
        background: linear-gradient(135deg, #f472b6 0%, #fb923c 100%) !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        color: #1e293b !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    [data-testid="metric-container"] {
        background: rgba(99, 102, 241, 0.15) !important;
        backdrop-filter: blur(10px) !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        border: 3px solid rgba(99, 102, 241, 0.3) !important;
    }

    /* Alert boxes - Softer colors */
    .stSuccess {
        background: linear-gradient(135deg, rgba(74, 222, 128, 0.25), rgba(74, 222, 128, 0.15)) !important;
        border-left: 8px solid #4ade80 !important;
        border-radius: 20px !important;
        padding: 1.8rem !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.25), rgba(248, 113, 113, 0.15)) !important;
        border-left: 8px solid #f87171 !important;
        border-radius: 20px !important;
        padding: 1.8rem !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.25), rgba(96, 165, 250, 0.15)) !important;
        border-left: 8px solid #60a5fa !important;
        border-radius: 20px !important;
        padding: 1.8rem !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Divider */
    hr {
        border: 0 !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(165, 180, 252, 0.5), transparent) !important;
        margin: 3rem 0 !important;
    }
    
    /* Hide branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 14px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(44, 62, 80, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #a5b4fc 0%, #c7d2fe 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #c7d2fe 0%, #a5b4fc 100%);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
    <div class="hero-section">
        <h1>🤖 AI-Powered Churn Prediction</h1>
        <p class="subtitle">Predict customer churn with advanced machine learning • Real-time insights • Data-driven decisions</p>
    </div>
""", unsafe_allow_html=True)

# --- Define File Paths ---
SCALER_FILE = "scaler.pkl"
MODEL_FILE = "xgboost_model.pkl"

# --- Helper Function ---
@st.cache_resource
def load_model(file_path):
    if not os.path.exists(file_path):
        st.error(f"⚠️ Model file not found: '{file_path}'")
        return None
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

# --- Load Models ---
with st.spinner("🔄 Loading AI models..."):
    scaler = load_model(SCALER_FILE)
    model = load_model(MODEL_FILE)

if not all([scaler, model]):
    st.warning("⚠️ Model loading failed. Please check your files.")
    st.stop()
else:
    st.success("✅ Models loaded successfully! Ready to predict.")

# --- Input Section Layout ---
st.markdown("### 📊 Enter Customer Information")
st.write("")

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container():
        st.markdown("#### 👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
        partner = st.selectbox("Has Partner?", ["No", "Yes"], key="partner")
        dependents_cat = st.selectbox("Has Dependents?", ["No", "Yes"], key="dependents")
        payment_method = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ], key="payment")

with col2:
    with st.container():
        st.markdown("#### 💼 Account Information")
        age = st.number_input("Age", min_value=18, max_value=120, value=30, key="age")
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, key="num_dep")
        num_referrals = st.number_input("Number of Referrals", min_value=0, max_value=50, value=0, key="referrals")
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12, key="tenure")
        monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=1000.0, value=50.00, format="%.2f", key="monthly")
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.00, format="%.2f", key="total")

# --- Prediction Section ---
st.markdown("<hr>", unsafe_allow_html=True)

# Fixed: Remove extra columns, just use the button directly
predict_clicked = st.button("🚀 Predict Customer Churn", use_container_width=True)

if predict_clicked:
    with st.spinner("🔮 Analyzing customer data..."):
        try:
            # Encoding
            gender_map = {'Male': 1, 'Female': 0}
            yes_no_map = {'Yes': 1, 'No': 0}
            payment_map = {
                "Electronic check": 0,
                "Mailed check": 1,
                "Bank transfer (automatic)": 2,
                "Credit card (automatic)": 3
            }

            cat_features = np.array([[gender_map[gender],
                                      yes_no_map[senior_citizen],
                                      yes_no_map[partner],
                                      yes_no_map[dependents_cat],
                                      payment_map[payment_method]]])

            num_features_unscaled = np.array([[age,
                                               num_dependents,
                                               num_referrals,
                                               tenure,
                                               monthly_charge,
                                               total_charges]])

            num_features_scaled = scaler.transform(num_features_unscaled)
            final_features = np.concatenate([cat_features, num_features_scaled], axis=1)

            prediction = model.predict(final_features)
            prediction_proba = model.predict_proba(final_features)

            result_class = prediction[0]
            proba_churn = prediction_proba[0][1]

            # Display results
            st.write("")
            st.write("")
            
            if result_class == 1:
                st.markdown("### 🚨 High Churn Risk Detected")
                st.error("**Prediction: Customer is likely to CHURN**")
                
                st.write("")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Churn Probability", f"{proba_churn * 100:.1f}%")
                with col_m2:
                    st.metric("Risk Level", "HIGH 🔴")
                with col_m3:
                    st.metric("Confidence", f"{max(proba_churn, 1-proba_churn) * 100:.1f}%")
                
                st.markdown("---")
                st.info("💡 **Recommendation**: Consider retention strategies such as personalized offers, loyalty programs, or customer support outreach.")
            else:
                st.markdown("### ✅ Low Churn Risk")
                st.success("**Prediction: Customer is likely to STAY**")
                
                st.write("")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Retention Probability", f"{(1 - proba_churn) * 100:.1f}%")
                with col_m2:
                    st.metric("Risk Level", "LOW 🟢")
                with col_m3:
                    st.metric("Confidence", f"{max(proba_churn, 1-proba_churn) * 100:.1f}%")
                
                st.markdown("---")
                st.info("💡 **Recommendation**: Continue current engagement strategies and maintain service quality to ensure long-term retention.")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("👆 Enter customer information above and click the **Predict** button to get AI-powered insights.")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: rgba(255, 255, 255, 0.85); font-size: 0.95rem; padding: 1.5rem 0;'>
        <p style='margin: 0 0 0.5rem 0; font-weight: 600; text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.4);'>
            🎯 Empowering Businesses with Intelligent Customer Retention
        </p>
        <p style='margin: 0; font-size: 0.85rem; color: rgba(255, 255, 255, 0.7); font-weight: 500;'>
            XGBoost ML Engine  • Real-time Analytics
        </p>
    </div>
""", unsafe_allow_html=True)