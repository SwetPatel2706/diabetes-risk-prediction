import json
import joblib
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clean medical-grade UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        color: #92400E;
        font-size: 0.92rem;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 1px solid #FDA4AF;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .risk-card-low {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #86EFAC;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .risk-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #E11D48;
    }
    .risk-low {
        color: #16A34A;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_metadata():
    try:
        pipeline = joblib.load("models/diabetes_pipeline.joblib")
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None, None
        
    try:
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}
        
    return pipeline, metadata


def main():
    pipeline, metadata = load_model_and_metadata()
    
    st.markdown('<div class="main-header">🩺 Diabetes Risk Screening Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An interactive clinical screening tool powered by a validated machine learning pipeline.</div>',
        unsafe_allow_html=True
    )
    
    # Prominent Medical Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical & Clinical Disclaimer:</strong><br>
        This application provides an estimated statistical risk based on the PIMA Indians Diabetes dataset methodology and is intended solely for preliminary screening demonstration and educational purposes. 
        <strong>It does NOT provide a medical diagnosis or treatment recommendation.</strong> Clinical decisions must always be made by licensed healthcare professionals utilizing comprehensive diagnostic evaluations (e.g., Fasting Plasma Glucose, HbA1c, OGTT).
    </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("Model artifact could not be found. Please ensure `models/diabetes_pipeline.joblib` exists.")
        return

    # Sidebar info & presets
    with st.sidebar:
        st.header("📋 Clinical Quick Presets")
        preset = st.selectbox(
            "Load Sample Patient Profile:",
            ["Custom Input", "Sample High-Risk Patient", "Sample Low-Risk Patient", "Sample Borderline Patient"],
            help="Choose a pre-filled profile to test model response or customize individual values below."
        )
        
        # Preset values matching real test points
        if preset == "Sample High-Risk Patient":
            def_preg, def_glu, def_bp, def_skin, def_ins, def_bmi, def_dpf, def_age = 8, 197, 74, 0, 0, 25.9, 1.191, 39
        elif preset == "Sample Low-Risk Patient":
            def_preg, def_glu, def_bp, def_skin, def_ins, def_bmi, def_dpf, def_age = 2, 74, 70, 20, 80, 22.0, 0.102, 22
        elif preset == "Sample Borderline Patient":
            def_preg, def_glu, def_bp, def_skin, def_ins, def_bmi, def_dpf, def_age = 5, 125, 78, 25, 110, 30.5, 0.450, 42
        else:
            # Default median-like baseline
            def_preg, def_glu, def_bp, def_skin, def_ins, def_bmi, def_dpf, def_age = 3, 117, 72, 23, 30, 32.0, 0.372, 29

        st.divider()
        st.subheader("Model Information")
        st.write("**Model Type:** Logistic Regression (Balanced)")
        st.write("**Cross-Validated AUC:** 0.836")
        st.write("**Decision Threshold:** 50.0% Risk")
        st.caption("Missing/zero clinical indicators (Insulin, SkinThickness, Glucose, Blood Pressure, BMI) are automatically imputed using training set medians within the pipeline.")

    # Form layout for clinical inputs
    st.subheader("Patient Clinical Measurements")
    st.write("Enter the patient's diagnostic values below. If a value is unknown or unrecorded, you may enter `0` (the pipeline treats `0` as missing and applies median imputation for clinical measurements).")

    col1, col2 = st.columns(2)

    with col1:
        glucose = st.number_input(
            "Plasma Glucose Concentration (2 hours in OGTT, mg/dL)",
            min_value=0, max_value=300, value=int(def_glu), step=1,
            help="Normal fasting level is usually < 100 mg/dL; postprandial normal is < 140 mg/dL. Enter 0 if unrecorded."
        )
        
        bmi = st.number_input(
            "Body Mass Index (BMI, weight in kg/(height in m)^2)",
            min_value=0.0, max_value=70.0, value=float(def_bmi), step=0.1, format="%.1f",
            help="Normal range: 18.5 - 24.9. Overweight: 25 - 29.9. Obese: >= 30. Enter 0.0 if unrecorded."
        )

        age = st.number_input(
            "Patient Age (years)",
            min_value=21, max_value=120, value=int(def_age), step=1,
            help="Dataset represents adult females aged 21 and older."
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function (DPF)",
            min_value=0.05, max_value=3.0, value=float(def_dpf), step=0.01, format="%.3f",
            help="Scores genetic/familial history risk of diabetes. Typical range is 0.08 to 2.4."
        )

    with col2:
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0, max_value=20, value=int(def_preg), step=1,
            help="Total number of times pregnant."
        )

        bp = st.number_input(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=0, max_value=160, value=int(def_bp), step=1,
            help="Normal diastolic pressure is typically < 80 mm Hg. Enter 0 if unrecorded."
        )

        insulin = st.number_input(
            "2-Hour Serum Insulin (mu U/ml)",
            min_value=0, max_value=900, value=int(def_ins), step=1,
            help="Normal 2h post-load insulin is roughly 16-166 mu U/ml. Enter 0 if unrecorded."
        )

        skin_thickness = st.number_input(
            "Triceps Skin Fold Thickness (mm)",
            min_value=0, max_value=100, value=int(def_skin), step=1,
            help="Measurement used to estimate subcutaneous body fat. Enter 0 if unrecorded."
        )

    # Prediction Action
    st.write("")
    calculate_btn = st.button("🔍 Assess Diabetes Risk", type="primary", use_container_width=True)

    if calculate_btn:
        # Prepare input dataframe matching exact feature names and order
        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])

        try:
            # Predict using pipeline
            risk_probability = pipeline.predict_proba(input_data)[0, 1]
            risk_percent = risk_probability * 100
            threshold = metadata.get("default_threshold", 0.5)
            is_high_risk = risk_probability >= threshold

            st.divider()
            st.subheader("Assessment Results")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if is_high_risk:
                    st.markdown(f"""
                    <div class="risk-card-high">
                        <h3 style="color: #9F1239; margin: 0;">HIGH RISK INDICATED</h3>
                        <div class="risk-value risk-high">{risk_percent:.1f}%</div>
                        <p style="color: #881337; margin: 0; font-weight: 500;">
                            Estimated Risk Probability (Cut-off: 50%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-card-low">
                        <h3 style="color: #14532D; margin: 0;">LOW RISK INDICATED</h3>
                        <div class="risk-value risk-low">{risk_percent:.1f}%</div>
                        <p style="color: #166534; margin: 0; font-weight: 500;">
                            Estimated Risk Probability (Cut-off: 50%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                st.markdown("#### Clinical Interpretation & Next Steps")
                if is_high_risk:
                    st.error(
                        f"**Flagged for Follow-up**: The estimated risk of **{risk_percent:.1f}%** meets or exceeds the 50% clinical screening threshold."
                    )
                    st.markdown("""
                    - **Recommended Action:** Schedule confirmatory diagnostic testing (Fasting Plasma Glucose, HbA1c, or Oral Glucose Tolerance Test).
                    - **Key Risk Drivers:** Elevated Glucose and BMI are the most influential positive drivers in this model.
                    """)
                else:
                    st.success(
                        f"**Routine Monitoring**: The estimated risk of **{risk_percent:.1f}%** is below the screening threshold."
                    )
                    st.markdown("""
                    - **Recommended Action:** Continue standard periodic screening and maintain a healthy lifestyle.
                    - **Note:** A negative screening result does not completely eliminate diabetes risk, especially if symptomatic.
                    """)

            # Feature Summary Table
            with st.expander("📊 View Input Summary & Preprocessing Imputation Details"):
                imputed_notes = []
                for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
                    val = input_data[col].iloc[0]
                    if val == 0:
                        med = metadata.get("imputed_medians", {}).get(col, "training median")
                        imputed_notes.append(f"- **{col}** entered as `0` → Imputed with training median: `{med}`")

                st.write("**Input Values Provided:**")
                st.dataframe(input_data, use_container_width=True)

                if imputed_notes:
                    st.info("**Automatic Imputations Applied:**\n" + "\n".join(imputed_notes))
                else:
                    st.caption("All clinical measurements provided as non-zero values; no missing value median imputation needed.")

        except Exception as err:
            st.error(f"An error occurred during risk calculation: {err}")


if __name__ == "__main__":
    main()
