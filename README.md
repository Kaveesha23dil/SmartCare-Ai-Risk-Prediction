# 🏥 SmartCare AI — Disease Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.0-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-EB5424.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **CCS3440 Artificial Intelligence Coursework Project**  
> An end-to-end Machine Learning pipeline and interactive clinical decision support system for **SmartCare Hospital** to classify and predict patient disease risk levels with Explainable AI (SHAP) and ethical fairness considerations.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
- [Dataset Overview](#-dataset-overview)
- [Machine Learning Workflow & Tasks](#-machine-learning-workflow--tasks)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Explainable AI (SHAP) & Ethics](#-explainable-ai-shap--ethics)
- [Interactive Streamlit Dashboard](#-interactive-streamlit-dashboard)
- [Getting Started](#-getting-started)
- [Notebooks Pipeline Execution](#-notebooks-pipeline-execution)
- [Technology Stack](#-technology-stack)
- [Project Team](#-project-team)
- [License](#-license)

---

## 🌟 Overview

The **SmartCare AI Risk Prediction System** is designed to assist medical practitioners at SmartCare Hospital in identifying high-risk and low-risk patient cases early. By combining robust preprocessing, state-of-the-art ML models, interpretable SHAP explanations, and a user-friendly Streamlit dashboard, this project bridges machine learning theory with real-world clinical decision support.

### 🔑 Core Capabilities
- **Comprehensive EDA & Preprocessing**: Handling missing clinical values, outlier detection, scaling, and feature interactions.
- **Multi-Model Development**: Training & tuning **Logistic Regression**, **Random Forest**, **XGBoost**, and **Decision Trees**.
- **Rigorous Evaluation**: Multi-metric evaluation focusing on **Macro F1**, **ROC-AUC**, and **Minority Class Recall** to prevent false negatives in patient care.
- **Explainability (XAI)**: SHAP summary, beeswarm, and waterfall plots providing transparent reasoning for individual patient risk scores.
- **AI Ethics & Fairness**: Audit of demographic bias, patient privacy, and clinical accountability guidelines.
- **Production-Ready Dashboard**: Interactive web application for real-time risk predictions with probability scores.

---

## 📁 Project Architecture & Directory Structure

```text
SmartCare-Ai-Risk-Prediction/
│
├── app/
│   └── app.py                                      # Streamlit web application
│
├── data/
│   ├── raw/
│   │   ├── smartcare_ai_dataset_1000.csv           # Original dataset (1,000 patient records)
│   │   └── smartcare_ai_dataset_data_dictionary.csv# Clinical data dictionary
│   ├── processed/
│   │   ├── smartcare_clean_dataset.csv             # Cleaned & feature-engineered dataset
│   │   ├── X_train.csv                             # Scaled training feature matrix
│   │   ├── X_test.csv                              # Scaled test feature matrix
│   │   ├── y_train.csv                             # Training ground truth labels
│   │   └── y_test.csv                              # Testing ground truth labels
│   └── README.md                                   # Data overview & pipeline documentation
│
├── models/
│   ├── logistic_regression.pkl                     # Best performing tuned model
│   ├── xgboost.pkl                                 # Tuned XGBoost classifier
│   ├── random_forest.pkl                           # Tuned Random Forest classifier
│   ├── decision_tree.pkl                           # Tuned Decision Tree classifier
│   ├── scaler.pkl                                  # Fitted feature scaler
│   ├── final_model_selection.json                  # Final selection criteria & winning model metadata
│   ├── model_metadata.json                         # Hyperparameters & CV scores for all models
│   ├── model_comparison.csv                        # Baseline vs. Tuned F1-Macro comparison
│   ├── per_class_metrics.csv                       # Class-specific precision, recall, and F1
│   ├── predictions/                                # Test set prediction probability CSVs
│   └── README.md                                   # Models directory notes
│
├── notebooks/
│   ├── 01_preprocessing_feature_engineering.ipynb  # Task 03 & 04: Data cleaning & feature engineering
│   ├── 02_exploratory_data_analysis.ipynb          # Task 02: EDA, distributions & correlation analysis
│   ├── 03_model_development.ipynb                  # Task 05: Model training, SMOTE & hyperparameter tuning
│   ├── 04_model_evaluation.ipynb                   # Task 06: Test evaluation, ROC curves & confusion matrices
│   ├── 05_explainable_ai_and_ethics.ipynb          # Tasks 07 & 08: SHAP XAI & AI ethics/fairness audit
│   └── 06_dashboard_deployment.ipynb               # Task 09: Streamlit dashboard integration & guide
│
├── reports/
│   ├── evaluation_results.csv                      # Full model benchmark comparison table
│   └── figures/                                    # Exported ROC curves, confusion matrices, SHAP plots
│
├── .gitignore                                      # Git ignore configuration
├── requirements.txt                                # Python package dependencies
├── LICENSE                                         # MIT License
└── README.md                                       # Main project documentation
```

---

## 📊 Dataset Overview

The dataset represents **1,000 patient records** containing physiological, demographic, and medical diagnostic indicators:

| Category | Features |
|---|---|
| **Demographic** | `Age`, `Gender`, `Socioeconomic_Status` |
| **Vital Signs** | `Systolic_BP`, `Diastolic_BP`, `Heart_Rate`, `BMI` |
| **Lab Measurements** | `Glucose_Level`, `Cholesterol_Total`, `HbA1c`, `Insulin` |
| **Lifestyle & History**| `Smoking_Status`, `Physical_Activity_Level`, `Family_History` |
| **Target Variable** | `Risk_Level` / `Outcome` (Multi-class / Binary Risk Classification) |

---

## 📓 Machine Learning Workflow & Tasks

| Step | Notebook / Module | Focus Areas & Deliverables |
|---|---|---|
| **Task 02** | [`02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb) | Univariate/Bivariate distributions, correlation heatmaps, class balance analysis |
| **Tasks 03 & 04** | [`01_preprocessing_feature_engineering.ipynb`](notebooks/01_preprocessing_feature_engineering.ipynb) | Missing value imputation, outlier handling, interaction terms, `StandardScaler` |
| **Task 05** | [`03_model_development.ipynb`](notebooks/03_model_development.ipynb) | Model training, SMOTE balancing, 5-Fold Cross Validation, GridSearchCV tuning |
| **Task 06** | [`04_model_evaluation.ipynb`](notebooks/04_model_evaluation.ipynb) | Test evaluation, ROC-AUC curves, confusion matrices, error analysis |
| **Tasks 07 & 08** | [`05_explainable_ai_and_ethics.ipynb`](notebooks/05_explainable_ai_and_ethics.ipynb) | SHAP Global/Local feature importance, bias audit, fairness, AI accountability |
| **Task 09** | [`06_dashboard_deployment.ipynb`](notebooks/06_dashboard_deployment.ipynb) & [`app/app.py`](app/app.py) | Full interactive Streamlit Dashboard prototype for clinical use |

---

## 🏆 Model Performance & Evaluation

All models were evaluated on an unseen, stratified test set. **Logistic Regression** achieved the highest overall generalizability and balance across metrics.

### 📈 Test Set Benchmark Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Macro ROC-AUC | Low-Class Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **Logistic Regression** | **0.930** | **0.947** | **0.920** | **0.932** | **0.930** | **0.976** | **0.885** |
| 🥈 **XGBoost** | 0.865 | 0.867 | 0.844 | 0.854 | 0.865 | 0.963 | 0.769 |
| 🥉 **Random Forest** | 0.790 | 0.823 | 0.738 | 0.767 | 0.788 | 0.913 | 0.577 |
| 🔹 **Decision Tree** | 0.730 | 0.737 | 0.733 | 0.734 | 0.729 | 0.798 | 0.731 |

> 📌 **Selected Best Model**: **Logistic Regression** (`models/logistic_regression.pkl`)  
> - **Macro F1 Score**: `0.9323`  
> - **Macro ROC-AUC**: `0.9759`  
> - **Low-Class Recall**: `88.46%` (Crucial for minimizing missed high/low risk anomalies)

---

## 🧠 Explainable AI (SHAP) & Ethics

### 🔍 Interpretability with SHAP
- **Global Feature Importance**: Identifies the primary drivers influencing model predictions across the entire patient population (e.g., Glucose, Blood Pressure, BMI).
- **Beeswarm Distribution**: Illustrates how higher or lower values of each biomarker shift risk probabilities.
- **Patient-Level Waterfall Plots**: Provides clinicians with a patient-specific breakdown of why a risk score was assigned.

### ⚖️ AI Ethics & Clinical Governance
- **Subgroup Fairness**: Evaluated error rates across age brackets and demographic subsets to prevent diagnostic disparities.
- **Human-in-the-Loop**: Designed strictly as an AI Decision Support Tool — clinicians retain final diagnostic authority.
- **Privacy Compliance**: Trained on de-identified records in accordance with healthcare data protection principles.

---

## 🖥️ Interactive Streamlit Dashboard

The web dashboard is located in [`app/app.py`](app/app.py) and features:
1. **🏠 Home**: Executive project summary and high-level architecture overview.
2. **🔍 Predict Risk**: Interactive patient input form with real-time risk classification and confidence probabilities.
3. **📊 Model Performance**: Interactive comparison tables and evaluation curves.
4. **🧠 Explainability**: SHAP global feature impact charts and interpretability insights.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/N3Edirisinghe/SmartCare-Ai-Risk-Prediction.git
cd SmartCare-Ai-Risk-Prediction
```

### 2️⃣ Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Launch the Streamlit App
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔄 Notebooks Pipeline Execution

To reproduce all experiments from scratch, run the notebooks in the following order:

```text
1. notebooks/02_exploratory_data_analysis.ipynb         # Explore distributions & patterns
2. notebooks/01_preprocessing_feature_engineering.ipynb # Generate data/processed/ splits
3. notebooks/03_model_development.ipynb                 # Train models & tune hyperparameters
4. notebooks/04_model_evaluation.ipynb                  # Compute metrics & generate curves
5. notebooks/05_explainable_ai_and_ethics.ipynb         # Run SHAP & ethics audits
6. notebooks/06_dashboard_deployment.ipynb              # Review dashboard deployment pipeline
```

---

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **Data Manipulation**: Pandas, NumPy, Scipy
- **Machine Learning**: Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
- **Model Explainability**: SHAP (SHapley Additive exPlanations)
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Web Deployment**: Streamlit
- **Model Persistence**: Joblib, JSON

---

## 👥 Project Team

| Student ID | Contributor | GitHub Profile | Primary Responsibility |
|:---:|:---|:---:|:---|
| **CIT-23-02-0021** | Nilupul Thisaranga | [@N3Edirisinghe](https://github.com/N3Edirisinghe) | Model Evaluation & Selection |
| **CIT-23-02-0025** | Siluna Nusal | [@GitGuru29](https://github.com/GitGuru29) | Explainable AI (XAI) & Prototype |
| **CIT-23-02-0042** | Dulani Madubashini | [@cobweb-sudo](https://github.com/cobweb-sudo) | Exploratory Data Analysis (EDA) |
| **CIT-23-02-0127** | Kaveesha Dilshan | [@Kaveesha23dil](https://github.com/Kaveesha23dil) | Data Preprocessing & Feature Engineering |
| **CIT-23-02-0359** | Zumra Hassan | [@Zumrahassan222](https://github.com/Zumrahassan222) | Model Development & Tuning |

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.
