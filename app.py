import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Sleep Health Predictor", layout="wide")

@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Handle Sleep Disorder (Target)
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("No Disorder")
    
    # 2. Split Blood Pressure
    bp = df["Blood Pressure"].astype(str).str.split("/", expand=True)
    df["Systolic"] = pd.to_numeric(bp[0], errors="coerce")
    df["Diastolic"] = pd.to_numeric(bp[1], errors="coerce")
    df.drop("Blood Pressure", axis=1, inplace=True)
    
    # 3. Fill Numeric NaNs
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Identify Categorical Columns
    categorical_cols = ['Gender', 'Occupation', 'BMI Category']
    target_col = 'Sleep Disorder'
    
    # Create Encoders for each column
    encoders = {}
    for col in categorical_cols + [target_col]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    return df, encoders

# Load data and prepare model/scaler
try:
    data_path = 'Sleep_health_and_lifestyle_daa.csv'
    df_clean, encoders = load_and_preprocess_data(data_path)
    
    # Define Features and Target
    X = df_clean.drop("Sleep Disorder", axis=1)
    y = df_clean["Sleep Disorder"]
    
    # Fit Scaler (matches notebook logic)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Model (Using Random Forest as per notebook)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)
    
except Exception as e:
    st.error(f"Error loading data: {e}")

# --- UI SECTION ---
st.title("🌙 Sleep Health & Lifestyle Predictor")
st.markdown("""
Predict the likelihood of **Sleep Apnea** or **Insomnia** based on your lifestyle metrics.
""")

with st.sidebar:
    st.header("User Input Features")
    
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    occupation = st.selectbox("Occupation", encoders['Occupation'].classes_)
    sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
    sleep_quality = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
    phys_activity = st.slider("Physical Activity Level (minutes/day)", 0, 120, 60)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    bmi_cat = st.selectbox("BMI Category", encoders['BMI Category'].classes_)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000)
    
    st.subheader("Blood Pressure")
    bp_sys = st.number_input("Systolic (top number)", min_value=80, max_value=200, value=120)
    bp_dia = st.number_input("Diastolic (bottom number)", min_value=40, max_value=130, value=80)

# --- PREDICTION LOGIC ---
if st.button("Predict Sleep Health"):
    # Create a dataframe for the input
    # Note: 'Person ID' is included as it was part of the training set in the notebook
    input_data = pd.DataFrame({
        'Person ID': [0], # Placeholder
        'Gender': [gender],
        'Age': [age],
        'Occupation': [occupation],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [sleep_quality],
        'Physical Activity Level': [phys_activity],
        'Stress Level': [stress_level],
        'BMI Category': [bmi_cat],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps],
        'Systolic': [bp_sys],
        'Diastolic': [bp_dia]
    })
    
    # Preprocess Inputs
    for col in ['Gender', 'Occupation', 'BMI Category']:
        input_data[col] = encoders[col].transform(input_data[col].astype(str))
    
    # Scale Inputs
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction_idx = model.predict(input_scaled)[0]
    prediction_label = encoders['Sleep Disorder'].inverse_transform([prediction_idx])[0]
    
    # Display Result
    st.divider()
    if prediction_label == "No Disorder":
        st.success(f"Result: **{prediction_label}** ✅")
        st.balloons()
    else:
        st.warning(f"Result: **{prediction_label}** ⚠️")
        st.info("Consider consulting a health professional for a detailed evaluation.")

# --- DATA VIEW ---
if st.checkbox("Show Sample Dataset"):
    st.write(pd.read_csv(data_path).head())