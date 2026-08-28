# Diabetes Risk Prediction Screening System

An end-to-end Machine Learning screening tool built to assess diabetes risk using clinical indicators from the PIMA Indians Diabetes dataset. The repository pairs research and experimentation in Jupyter with an interactive web application built in Streamlit.

---

## 🎯 Project Objective

Early detection of diabetes significantly reduces long-term complications such as cardiovascular disease, neuropathy, and retinopathy. This project aims to build a clinical screening model that:
1. Accurately identifies patients with a high risk of diabetes.
2. Prioritizes **recall** on positive diabetic cases to minimize missed diagnoses.
3. Provides explainable, calibrated risk probabilities rather than uninterpretable predictions.
4. Delivers an accessible web interface for preliminary clinical screening.

---

## 📊 Dataset Overview

- **Source:** [PIMA Indians Diabetes Database](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv) (CC0 Public Domain).
- **Cohort:** 768 female patients of Pima Indian heritage aged 21 years and older.
- **Class Distribution:** 500 non-diabetic (65.1%) vs. 268 diabetic (34.9%) — moderate class imbalance.

### Features
| Feature Name | Description | Units |
| :--- | :--- | :--- |
| `Pregnancies` | Number of times pregnant | Count |
| `Glucose` | Plasma glucose concentration (2 hours in oral glucose tolerance test) | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skin fold thickness | mm |
| `Insulin` | 2-Hour serum insulin | $\mu\text{U/mL}$ |
| `BMI` | Body mass index | $\text{kg/m}^2$ |
| `DiabetesPedigreeFunction` | Genetic score indicating diabetes family history | Score |
| `Age` | Age in years | Years |
| **`Outcome`** | **Target Variable** (0 = No Diabetes, 1 = Diabetes) | Binary |

---

## 🔬 Machine Learning Methodology & Preprocessing

The methodology strictly preserves the research workflow developed in [`notebooks/diabetes_risk_prediction.ipynb`](notebooks/diabetes_risk_prediction.ipynb):

### 1. Zero-as-Missing Value Handling
In clinical measurements, values of `0` for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are physiologically impossible and represent missing/unrecorded entries (e.g., Insulin is unrecorded for 374 out of 768 patients).
- These zeroes are identified and converted to `NaN`.
- `Pregnancies` and `DiabetesPedigreeFunction` are retained as valid numeric quantities.

### 2. Leakage-Free Train/Test Split
- The dataset is split into **80% training (614 patients)** and **20% testing (154 patients)** using `stratify=y` (`random_state=42`) to maintain class ratios.
- **Median Imputation:** Missing values in the 5 clinical columns are filled using the **training set medians only** (`Glucose: 117`, `BloodPressure: 72`, `SkinThickness: 29`, `Insulin: 125`, `BMI: 32.3`). This prevents data leakage from the test set.

### 3. Feature Scaling
- Features are normalized using `StandardScaler` fitted on the training split so that high-magnitude features (e.g., Insulin) do not dominate linear coefficients.

---

## 📈 Model Evaluation & Selection

Three candidate architectures were evaluated using **5-Fold Stratified Cross-Validation** on ROC-AUC:

| Model | 5-Fold Stratified CV ROC-AUC | Interpretability | Inference Speed |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **0.836 (± 0.046)** | **High (Direct Odds/Coefficients)** | **Instant** |
| Random Forest (Balanced) | 0.825 (± 0.054) | Moderate | Fast |
| XGBoost (Scale Pos Weight) | 0.819 (± 0.055) | Low (Black-box) | Fast |

### Why Logistic Regression Was Selected:
1. **Best Generalization:** Delivered the highest mean cross-validated AUC (0.836) with the lowest variance.
2. **Clinical Interpretability:** The model provides direct coefficients indicating feature importance:
   - **`Glucose` (+1.18)** and **`BMI` (+0.71)** are the strongest risk drivers, aligning with medical science.
3. **Calibrated Probabilities:** Predicts true probabilistic risk scores that allow clinics to adjust operational decision thresholds based on screening capacity.
4. **Parsimony:** Simple, robust against overfitting on a 768-sample dataset.

### Test Set Performance (50% Decision Cut-off)
- **Test ROC-AUC:** 0.813
- **Test Recall (Diabetes):** 70% (catches 38 out of 54 true cases)
- **Test Precision (Diabetes):** 60%

---

## 📁 Repository Structure

```
diabetes-risk-prediction/
├── data/
│   └── diabetes.csv                     # Raw PIMA dataset
├── models/
│   ├── diabetes_pipeline.joblib         # Serialized scikit-learn pipeline
│   └── model_metadata.json              # Trained parameters, medians & metrics
├── notebooks/
│   └── diabetes_risk_prediction.ipynb   # Research, EDA, training & evaluation
├── app.py                               # Interactive Streamlit application
├── train_pipeline.py                    # Script to train & export pipeline artifact
├── requirements.txt                     # Project dependencies
├── README.md                            # Comprehensive project documentation
└── .gitignore                           # Git ignore rules
```

---

## 🚀 Installation & Usage

### 1. Clone & Navigate
```bash
git clone https://github.com/SwetPatel2706/diabetes-risk-prediction.git
cd diabetes-risk-prediction
```

### 2. Set Up & Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# On Windows (Command Prompt / PowerShell):
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Model Pipeline
```bash
python train_pipeline.py
```

### 4. Launch the Streamlit App
Make sure your virtual environment is activated (`source .venv/bin/activate`):
```bash
streamlit run app.py
```
*Tip: You can also run directly without activating:*
```bash
.venv/bin/streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚠️ Limitations & Medical Disclaimer

- **Screening Tool, Not a Diagnosis:** This software calculates statistical risk and must not be used as a substitute for professional medical advice, clinical diagnosis, or lab testing.
- **Demographic Scope:** The training cohort consists exclusively of adult females of Pima Indian ancestry aged 21 and older. Generalization to other demographics, males, or pediatric cohorts requires external clinical validation.
- **Missing Insulin Data:** ~48% of the cohort lacked insulin measurements, requiring median imputation.
