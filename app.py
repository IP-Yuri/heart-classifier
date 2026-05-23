import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# matplotlib.use('Agg') # Set backend to non-interactive for server stability (Removed to allow pop-up plots)

#machine learning model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, classification_report
import joblib

#interface
import streamlit as st

@st.cache_resource
def run_ml_pipeline():
    #================================================================================
    #PHASE 1: EDA and Data cleaning
    # ================================================================================

    df = pd.read_csv('heart_disease_uci.csv')

    #data overview
    print("Dataset Overview:")
    print(df.head())
    print(df.info())


    #display missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])  #only missing values > 0 will be displayed

    #visualizing sick vs healthy population
    plt.figure(figsize=(6,6))
    sns.countplot(x='num', data=df)
    plt.xlabel('Disease Presence')            #0 = healthy, >0 = heart disease presence
    plt.ylabel('Count')
    plt.title('Sick vs Healthy Population')
    plt.show()

    #handling missing values
    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    #we fill the numerical gaps with median 
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    #we fill the categorical gaps with mode
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    #verify missing values are gone
    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    # ============================================================================
    # PHASE 2:  DATA PREPARATION FOR MACHINE LEARNING MODEL TRAINING
    # ============================================================================

    #encoding categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    #isolating features and target variable
    X = df.drop(['num', 'id'], axis=1)
    y = df['num']

    y= np.where(y>0, 1, 0)    #ensure y is only 1 and 0 (sick/healthy)

    #we split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)    #stratify ensures the class distribution is preserved in both train and test sets

    #scaling numerical variables
    scaler = StandardScaler()         
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #model training
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    #===============================================================================================
    #PHASE 3: MODEL EVALUATION
    #===============================================================================================

    #make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    #metrics output 
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    cls_rep = classification_report(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", acc)
    print("Recall:", rec)
    print("Precision:", prec)
    print("Classification Report:\n", cls_rep)

    #visualization of predicted vs actual values
    plt.figure(figsize=(10,6))
    sns.countplot(x='num', data=pd.DataFrame({'num': y_test}))
    plt.title('Actual Heart Disease Distribution')
    plt.xlabel('Disease Presence (0: Healthy, 1: Diseased)')
    plt.ylabel('Count')
    plt.show()

    # ============================================================================
    # PHASE 4: MODEL EXPORT
    # ============================================================================

    # Saving the trained model and scaler for future use
    joblib.dump(model, 'heart_disease_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')   #saving feature names for reference

    return model, scaler, X.columns.tolist()

# Load model and components via the cached pipeline
model, scaler, feature_names = run_ml_pipeline()

# ============================================================================
# PHASE 5: USER INTERFACE (Cyber-Luxury Aesthetic)
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
