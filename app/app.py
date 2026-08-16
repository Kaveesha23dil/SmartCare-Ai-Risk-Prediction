"""
SmartCare AI Risk Prediction - Streamlit Dashboard
CCS3440 Artificial Intelligence Coursework
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import shap

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCare AI Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Helper: Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str):
    model_path = os.path.join("models", f"{model_name}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_model_selection():
    path = os.path.join("models", "final_model_selection.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=80)
    st.title("SmartCare AI")
    st.caption("Disease Risk Prediction System")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Predict Risk", "📊 Model Performance", "🧠 Explainability"],
    )

    st.divider()
    model_info = load_model_selection()
    if model_info:
        st.success(f"✅ Best Model: **{model_info.get('best_model', 'N/A')}**")
        st.metric("ROC-AUC", model_info.get("roc_auc", "N/A"))

# ─────────────────────────────────────────────
# Home Page
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏥 SmartCare AI — Disease Risk Prediction")
    st.markdown(
        """
        Welcome to the **SmartCare AI Risk Prediction Dashboard**.  
        This system uses machine learning models trained on patient data to classify disease risk.

        ---
        ### 🔑 Key Features
        - **4 ML Models**: Logistic Regression, Random Forest, XGBoost, Decision Tree
        - **Explainable AI**: SHAP-based feature importance for transparent decisions
        - **Real-time Prediction**: Enter patient data and get instant risk classification
        - **Ethical AI**: Bias analysis and fairness evaluation included

        ---
        ### 📋 How to Use
        1. Navigate to **Predict Risk** to input patient data
        2. View **Model Performance** for evaluation metrics
        3. Explore **Explainability** for SHAP-based insights
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Models Trained", "4")
    col2.metric("Task", "Classification")
    col3.metric("Explainability", "SHAP")

# ─────────────────────────────────────────────
# Predict Risk Page
# ─────────────────────────────────────────────
elif page == "🔍 Predict Risk":
    st.title("🔍 Patient Risk Prediction")
    st.info("Enter patient details below to predict disease risk.")

    model_choice = st.selectbox(
        "Select Model",
        ["random_forest", "xgboost", "logistic_regression", "decision_tree"],
    )

    with st.form("prediction_form"):
        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)

        with col2:
            blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

        with col3:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)

        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

    if submitted:
        model = load_model(model_choice)
        if model is None:
            st.warning("⚠️ Model not found. Please train and save the model first.")
        else:
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]])
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.divider()
            col1, col2 = st.columns(2)
            if prediction == 1:
                col1.error(f"⚠️ **High Risk** — Disease Risk Detected")
            else:
                col1.success(f"✅ **Low Risk** — No Significant Disease Risk")
            col2.metric("Risk Probability", f"{probability:.2%}")

# ─────────────────────────────────────────────
# Model Performance Page
# ─────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.title("📊 Model Evaluation Results")

    results_path = os.path.join("reports", "evaluation_results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        st.dataframe(df, use_container_width=True)

        fig_path = os.path.join("reports", "figures")
        if os.path.exists(fig_path):
            images = [f for f in os.listdir(fig_path) if f.endswith((".png", ".jpg"))]
            if images:
                st.subheader("📈 Evaluation Figures")
                cols = st.columns(2)
                for i, img in enumerate(images):
                    cols[i % 2].image(os.path.join(fig_path, img), use_column_width=True)
    else:
        st.warning("⚠️ No evaluation results found. Please run the model evaluation notebook first.")

# ─────────────────────────────────────────────
# Explainability Page
# ─────────────────────────────────────────────
elif page == "🧠 Explainability":
    st.title("🧠 Explainable AI — SHAP Analysis")
    st.markdown(
        """
        SHAP (SHapley Additive exPlanations) values explain the contribution of each feature
        to the model's prediction, enabling transparent and trustworthy AI for medical staff.
        """
    )

    shap_path = os.path.join("reports", "figures")
    if os.path.exists(shap_path):
        shap_figs = [f for f in os.listdir(shap_path) if "shap" in f.lower()]
        if shap_figs:
            for fig in shap_figs:
                st.image(os.path.join(shap_path, fig), use_column_width=True)
        else:
            st.info("ℹ️ No SHAP plots found. Run `05_explainable_ai_and_ethics.ipynb` to generate them.")
    else:
        st.info("ℹ️ Reports directory not found.")
