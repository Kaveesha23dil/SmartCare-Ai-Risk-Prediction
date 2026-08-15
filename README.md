# 🏥 SmartCare AI — Disease Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Prototype-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Empty_Scaffold-lightgrey.svg)](#-project-status)

> **CCS3440 Artificial Intelligence Coursework Project**  
> An empty project scaffold for **Option C — Disease Risk Classification**, prepared for an end-to-end machine-learning pipeline and clinical decision-support prototype.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Status](#-project-status)
- [Project Structure](#-project-architecture--directory-structure)
- [Dataset Overview](#-dataset-overview)
- [Machine Learning Workflow](#-machine-learning-workflow--tasks)
- [Evaluation Plan](#-model-evaluation-plan)
- [Explainable AI and Ethics](#-explainable-ai--ethics)
- [Streamlit Prototype](#-interactive-streamlit-prototype)
- [Getting Started](#-getting-started)
- [Notebook Execution Order](#-notebook-pipeline-execution)
- [Technology Stack](#-technology-stack)
- [Project Team](#-project-team)
- [Team Branch Workflow](#-team-branch-workflow)
- [Academic Integrity](#-academic-integrity)

---

## 🌟 Overview

The planned **SmartCare AI Disease Risk Prediction System** will classify patients into three disease-risk categories:

- `Low`
- `Medium`
- `High`

The coursework target variable is `disease_risk_level`. The completed project will cover data preparation, exploratory analysis, feature engineering, model comparison, multiclass evaluation, explainable AI, and an interactive prototype.

> [!IMPORTANT]
> This project is for education and decision support. A prediction must not be treated as a medical diagnosis or replace qualified clinical judgment.

---

## 🚧 Project Status

This repository contains **empty files and folders only**, apart from this README and the existing Git configuration.

| Component | Status |
|---|:---:|
| Reference repository structure | ✅ Mirrored |
| README documentation | ✅ Created |
| Dataset contents | ⬜ Empty placeholders |
| Notebook contents | ⬜ Empty placeholders |
| Application code | ⬜ Empty placeholder |
| Trained models | ⬜ Empty placeholders |
| Metrics and predictions | ⬜ Empty placeholders |
| License and dependencies | ⬜ Empty placeholders |

No model results, dataset values, implementation code, or research findings have been copied from the reference repository.

---

## 📁 Project Architecture & Directory Structure

```text
SmartCare-Ai-Risk-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── processed/
│   │   ├── .gitkeep
│   │   ├── smartcare_clean_dataset.csv
│   │   ├── X_test.csv
│   │   ├── X_train.csv
│   │   ├── y_test.csv
│   │   └── y_train.csv
│   ├── raw/
│   │   ├── .gitkeep
│   │   ├── smartcare_ai_dataset_1000.csv
│   │   └── smartcare_ai_dataset_data_dictionary.csv
│   └── README.md
│
├── models/
│   ├── predictions/
│   │   ├── decision_tree_test_predictions.csv
│   │   ├── logistic_regression_test_predictions.csv
│   │   ├── random_forest_test_predictions.csv
│   │   └── xgboost_test_predictions.csv
│   ├── .gitkeep
│   ├── decision_tree.pkl
│   ├── evaluation_results.csv
│   ├── final_model_selection.json
│   ├── logistic_regression.pkl
│   ├── model_comparison.csv
│   ├── model_metadata.json
│   ├── per_class_metrics.csv
│   ├── random_forest.pkl
│   ├── README.md
│   ├── scaler.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   ├── 01_preprocessing_feature_engineering.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_explainable_ai_and_ethics.ipynb
│   └── 06_dashboard_deployment.ipynb
│
├── reports/
│   ├── figures/
│   │   └── .gitkeep
│   └── evaluation_results.csv
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Every path above matches the referenced repository. Files intended to hold datasets, notebooks, models, results, predictions, application code, dependencies, or license text are deliberately empty.

---

## 📊 Dataset Overview

The coursework provides:

- `smartcare_ai_dataset_1000.csv`
- `smartcare_ai_dataset_data_dictionary.csv`

The dataset contains 1,000 hospital records covering patient, clinical, operational, and financial information.

| Category | Examples from the coursework |
|---|---|
| **Patient information** | Patient ID, age, gender, blood group |
| **Clinical information** | Diagnosis, blood pressure, blood sugar, cholesterol, BMI |
| **Hospital operations** | Department, appointments, admissions, stay, room, treatments, laboratory tests |
| **Financial information** | Consultation, laboratory, room, medicine, and total charges |
| **Target variable** | `disease_risk_level` |
| **Classes** | `Low`, `Medium`, `High` |

The empty raw-data placeholders must be replaced with the lecturer-provided files. No external dataset is required.

> [!CAUTION]
> Do not publish healthcare-related data without authorization. Remove identifiers from notebooks, screenshots, logs, and reports.

---

## 📓 Machine Learning Workflow & Tasks

| Coursework task | Planned file/location | Expected work |
|---|---|---|
| **Problem Definition & Literature Review** | Technical report | Define the problem and review at least five peer-reviewed papers |
| **Dataset Understanding** | `02_exploratory_data_analysis.ipynb` | Attributes, quality, statistics, distributions, correlations |
| **Preprocessing & Feature Engineering** | `01_preprocessing_feature_engineering.ipynb` | Missing values, duplicates, outliers, encoding, scaling, selection |
| **Model Development** | `03_model_development.ipynb` | Train and tune at least three classifiers |
| **Model Evaluation** | `04_model_evaluation.ipynb` | Metrics, confusion matrices, comparison, final selection |
| **Explainable AI & Ethics** | `05_explainable_ai_and_ethics.ipynb` | SHAP/LIME/importance, transparency, fairness, limitations |
| **Prototype Development** | `06_dashboard_deployment.ipynb`, `app/app.py` | Patient input, processing, prediction, result display |

---

## 🏆 Model Evaluation Plan

The coursework requires multiclass:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

At least three models should be compared. Possible algorithms include Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, Naive Bayes, K-Nearest Neighbors, and optional XGBoost.

No best model or performance value is claimed until experiments are completed on an unseen test set.

---

## 🧠 Explainable AI & Ethics

The final solution must use **SHAP**, **LIME**, or **feature importance analysis** to discuss:

- Important predictive features
- Individual prediction reasoning
- Model transparency
- Ethical implications
- Privacy and demographic bias
- Limitations and clinical accountability

Healthcare predictions should remain human-supervised and should never be presented as diagnostic certainty.

---

## 🖥️ Interactive Streamlit Prototype

The planned prototype in `app/app.py` should:

1. Accept validated patient information.
2. Apply the fitted preprocessing workflow.
3. Generate a `Low`, `Medium`, or `High` prediction.
4. Display the result clearly and responsibly.

The application file is currently empty.

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone <your-repository-url>
cd SmartCare-Ai-Risk-Prediction
```

### 2️⃣ Create a virtual environment

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Add the coursework dataset

Replace the empty CSV placeholders under `data/raw/` with the lecturer-provided files.

### 4️⃣ Add and install dependencies

Populate `requirements.txt` only after selecting and pinning the required package versions.

---

## 🔄 Notebook Pipeline Execution

After the notebooks are implemented, run them in this order:

```text
1. notebooks/01_preprocessing_feature_engineering.ipynb
2. notebooks/02_exploratory_data_analysis.ipynb
3. notebooks/03_model_development.ipynb
4. notebooks/04_model_evaluation.ipynb
5. notebooks/05_explainable_ai_and_ethics.ipynb
6. notebooks/06_dashboard_deployment.ipynb
```

---

## 🛠️ Technology Stack

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Optional Machine Learning:** XGBoost, TensorFlow, Keras
- **Explainability:** SHAP or LIME
- **Prototype:** Streamlit or Flask
- **Model Persistence:** Pickle or Joblib

---

## 👥 Project Team

| Student ID | Contributor | GitHub Profile | Primary Responsibility | Assigned Branch |
|:---:|---|:---:|---|:---:|
| **CIT-23-02-0021** | Nilupul Thisaranga | [@N3Edirisinghe](https://github.com/N3Edirisinghe) | Model Evaluation & Selection | `member-01` |
| **CIT-23-02-0025** | Siluna Nusal | [@GitGuru29](https://github.com/GitGuru29) | Explainable AI (XAI) & Prototype | `member-02` |
| **CIT-23-02-0042** | Dulani Madubashini | [@cobweb-sudo](https://github.com/cobweb-sudo) | Exploratory Data Analysis (EDA) | `member-03` |
| **CIT-23-02-0127** | Kaveesha Dilshan | [@Kaveesha23dil](https://github.com/Kaveesha23dil) | Data Preprocessing & Feature Engineering | `member-04` |
| **CIT-23-02-0359** | Zumra Hassan | [@Zumrahassan222](https://github.com/Zumrahassan222) | Model Development & Tuning | `member-05` |

> [!WARNING]
> The supplied team list contains five students, but the coursework specification states a maximum of four students per group. Confirm the approved group size with the lecturer before submission.

---

## 🌿 Team Branch Workflow

Each member must work only on their assigned predefined branch:

| Member | Branch |
|:---:|---|
| Member 01 | `member-01` |
| Member 02 | `member-02` |
| Member 03 | `member-03` |
| Member 04 | `member-04` |
| Member 05 | `member-05` |

### First-time branch setup

```bash
git fetch origin
git switch <assigned-branch>
git push -u origin <assigned-branch>
```

### Normal contribution workflow

```bash
git switch <assigned-branch>
git pull --rebase origin <assigned-branch>
git add .
git commit -m "Describe the completed work"
git push origin <assigned-branch>
```

Members must not push coursework changes directly to `main`. Completed work should be reviewed through a pull request from the assigned member branch into `main`. Before opening a pull request, synchronize the member branch with the latest `main` and resolve conflicts locally.

---

## 🎓 Academic Integrity

All analysis, code, results, and writing must be the group's own work. Every member must understand and be able to explain the data, preprocessing, algorithms, metrics, explainability outputs, prototype, and individual contribution during the viva.

---

<p align="center">
  <strong>CCS3440 Artificial Intelligence Coursework — Option C</strong><br>
  Disease Risk Classification for SmartCare Hospital
</p>
