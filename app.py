import pandas as pd
import streamlit as st
import joblib

# ============================================================================
# PHASE 1: LOAD PRE-TRAINED MODEL
# ============================================================================

# Load model and components directly from your repo files
@st.cache_resource
def load_components():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_components()

# ============================================================================
# PHASE 2: USER INTERFACE (Cyber-Luxury Aesthetic)
# ============================================================================

# Page Configuration
st.set_page_config(page_title="HEART SCAN v1.0", layout="wide")

# Cyber-Luxury Styling
st.markdown("""
    <style>
    .main {
        background-color: #050505;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #404040;
        border-radius: 0px;
        font-weight: bold;
        letter-spacing: 2px;
        height: 3em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #ffffff;
    }
    h1 {
        letter-spacing: -2px;
        font-weight: 800;
        border-bottom: 4px solid #ffffff;
        padding-bottom: 10px;
        margin-bottom: 30px;
        text-transform: uppercase;
    }
    .stNumberInput, .stSelectbox {
        border-radius: 0px;
    }
    div[data-baseweb="select"] {
        border-radius: 0px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("HEART SCAN v1.0")

# Symmetrical Layout
col_num, col_cat = st.columns(2)

with col_num:
    st.subheader("BIOMETRIC DATA")
    age = st.number_input("AGE", min_value=1, max_value=120, value=50)
    trestbps = st.number_input("RESTING BLOOD PRESSURE (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("SERUM CHOLESTEROL (mg/dl)", min_value=100, max_value=600, value=200)
    thalch = st.number_input("MAX HEART RATE ACHIEVED", min_value=50, max_value=250, value=150)
    oldpeak = st.number_input("ST DEPRESSION (OLDPEAK)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ca = st.number_input("MAJOR VESSELS (0-4)", min_value=0, max_value=4, value=0)

with col_cat:
    st.subheader("CLINICAL PROFILES")
    sex = st.selectbox("SEX", options=["Male", "Female"])
    dataset = st.selectbox("ORIGIN DATASET", options=["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
    cp = st.selectbox("CHEST PAIN TYPE", options=["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    fbs = st.selectbox("FASTING BLOOD SUGAR > 120 mg/dl", options=[False, True])
    restecg = st.selectbox("RESTING ECG RESULTS", options=["normal", "st-t abnormality", "lv hypertrophy"])
    exang = st.selectbox("EXERCISE INDUCED ANGINA", options=[False, True])
    slope = st.selectbox("PEAK EXERCISE SLOPE", options=["upsloping", "flat", "downsloping"])
    thal = st.selectbox("THALLIUM TEST", options=["normal", "fixed defect", "reversable defect"])

# THE BRIDGE: Prediction Logic
st.markdown("---")
if st.button("INITIALIZE NEURAL SCAN"):
    # 1. Create Input DataFrame matching feature_names
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # 2. Map Numerical Values
    input_df['age'] = age
    input_df['trestbps'] = trestbps
    input_df['chol'] = chol
    input_df['thalch'] = thalch
    input_df['oldpeak'] = oldpeak
    input_df['ca'] = ca
    
    # 3. Map Categorical Values (Handling drop_first=True)
    if sex == "Male": input_df['sex_Male'] = 1
    if dataset == "Hungary": input_df['dataset_Hungary'] = 1
    if dataset == "Switzerland": input_df['dataset_Switzerland'] = 1
    if dataset == "VA Long Beach": input_df['dataset_VA Long Beach'] = 1
    if cp == "atypical angina": input_df['cp_atypical angina'] = 1
    if cp == "non-anginal": input_df['cp_non-anginal'] = 1
    if cp == "typical angina": input_df['cp_typical angina'] = 1
    if fbs == True: input_df['fbs_True'] = 1
    if restecg == "normal": input_df['restecg_normal'] = 1
    if restecg == "st-t abnormality": input_df['restecg_st-t abnormality'] = 1
    if exang == True: input_df['exang_True'] = 1
    if slope == "flat": input_df['slope_flat'] = 1
    if slope == "upsloping": input_df['slope_upsloping'] = 1
    if thal == "normal": input_df['thal_normal'] = 1
    if thal == "reversable defect": input_df['thal_reversable defect'] = 1
    
    # 4. Scale Input
    input_scaled = scaler.transform(input_df)
    
    # 5. Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # 6. Output
    st.subheader("DIAGNOSTIC VERDICT")
    if prediction == 1:
        st.error(f"WARNING: POTENTIAL CARDIAC PATHOLOGY DETECTED (Probability: {probability:.2%})")
        st.write("Diagnostic confidence indicates high correlation with ischemic heart disease markers.")
    elif probability > 0.3:
        st.warning(f"CAUTION: ELEVATED RISK FACTORS DETECTED (Probability: {probability:.2%})")
        st.write("Patient exhibits borderline biometric markers. Further clinical investigation recommended.")
    else:
        st.success(f"STATUS: CARDIAC PROFILE WITHIN OPTIMAL RANGE (Probability: {probability:.2%})")
        st.write("Neural analysis indicates low probability of coronary artery disease.")