import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_and_save():
    # 1. Load data
    df = pd.read_csv("data/diabetes.csv")
    
    zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    passthrough_cols = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]
    
    # Preserve full feature order
    feature_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    target_col = "Outcome"
    
    X = df[feature_cols]
    y = df[target_col]

    # 2. Stratified train/test split matching notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Standard scikit-learn ColumnTransformer:
    # Impute 0 values with median on the 5 clinical columns, passthrough others
    preprocessor = ColumnTransformer(
        transformers=[
            ("impute_zeros", SimpleImputer(missing_values=0, strategy="median"), zero_as_missing_cols),
            ("passthrough", "passthrough", passthrough_cols)
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])

    # 4. Fit Pipeline on training data
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on test data
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("Test Confusion Matrix:")
    print(cm)
    print("\nROC AUC:", round(auc, 3))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    # 6. Verify sample predictions matching notebook
    print("\n--- Verifying sample predictions matching notebook ---")
    for i in [7, 4, 3]:
        patient_df = X_test.iloc[[i]]
        pred = pipeline.predict(patient_df)[0]
        prob = pipeline.predict_proba(patient_df)[0, 1]
        print(f"Patient index {i} in test set:")
        print(f"  Inputs: {patient_df.to_dict(orient='records')[0]}")
        print(f"  Predicted: {'Diabetes' if pred == 1 else 'No Diabetes'} (Risk: {prob*100:.1f}%) | Actual: {y_test.iloc[i]}")

    # Extract learned medians
    imputer = pipeline.named_steps["preprocessor"].named_transformers_["impute_zeros"]
    imputed_medians = dict(zip(zero_as_missing_cols, [float(m) for m in imputer.statistics_]))

    # 7. Save pipeline artifact and metadata
    model_path = "models/diabetes_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved pipeline to {model_path}")

    metadata = {
        "features": feature_cols,
        "zero_as_missing_features": zero_as_missing_cols,
        "passthrough_features": passthrough_cols,
        "target": target_col,
        "imputed_medians": imputed_medians,
        "model_type": "LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)",
        "default_threshold": 0.50,
        "test_roc_auc": round(float(auc), 3)
    }
    
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to models/model_metadata.json")


if __name__ == "__main__":
    train_and_save()
