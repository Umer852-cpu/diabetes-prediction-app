"""
app.py — Diabetes Risk Prediction Web App (Streamlit)  (v4 - leakage-free)

Uses the same WeightedHybrid class as the research pipeline (diabetes_pipeline.py):
  * cleaning statistics (medians, outlier limits) learned from TRAINING data only
  * hybrid weights learned from out-of-fold training predictions (test set never used)
  * SHAP explains the hybrid model itself for each patient

Run locally:   streamlit run app.py
Deploy free:   push to GitHub, then connect on streamlit.io (see DEPLOY_STEPS.md)
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

from diabetes_pipeline import WeightedHybrid, scores, BASE_NAMES, SEED

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="centered")


# ----------------------------------------------------------------------
# Train everything ONCE and cache it (so the app stays fast after first load)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Model training ho raha hai (sirf pehli baar hota hai)...")
def train_everything():
    df = pd.read_csv("data/diabetes_raw.csv")
    X, y = df.drop(columns="Outcome"), df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    hybrid = WeightedHybrid().fit(X_train, y_train)          # no test data used here

    base_probs = hybrid.base_probabilities(X_test)
    base_probs["Hybrid Model"] = hybrid.predict_proba(X_test)
    test_scores = {n: scores(y_test, p) for n, p in base_probs.items()}

    background = hybrid.prep.transform(X_train).sample(100, random_state=SEED)
    explainer = shap.PermutationExplainer(hybrid.predict_proba_processed, background, seed=SEED)
    return hybrid, test_scores, explainer


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🩺 Diabetes Risk Prediction")
st.caption("Hybrid Ensemble (Logistic Regression + Decision Tree + Random Forest + XGBoost) + SHAP Explainability")
st.caption("Muhammad Umer Hanif | 303-241031 | BS AI, Hazara University Mansehra")

hybrid, test_scores, explainer = train_everything()

with st.expander("ℹ️ Model performance (held-out test set, 154 patients)"):
    perf = pd.DataFrame(test_scores).T[["Accuracy", "ROC-AUC"]].round(3)
    perf["Hybrid Weight"] = [round(hybrid.weights.get(n, float("nan")), 3) for n in perf.index]
    st.dataframe(perf, use_container_width=True)
    st.caption("Weights are learned from out-of-fold predictions on the training data only.")

st.subheader("Patient Information")
st.caption("Enter 0 for Skin Thickness or Insulin if not measured — the training median will be used.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose (mg/dL)", 40, 300, 120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 30, 200, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI", 10.0, 70.0, 28.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.05, 2.5, 0.4)
    age = st.number_input("Age", 15, 100, 30)

if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
    patient = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }])
    X_proc = hybrid.prep.transform(patient)                  # same cleaning as training
    prob = float(hybrid.predict_proba_processed(X_proc)[0])

    prediction = "YES — Diabetic" if prob >= 0.5 else "NO — Not Diabetic"
    risk_level = "LOW RISK" if prob < 0.33 else "MODERATE RISK" if prob < 0.66 else "HIGH RISK"
    color = "🟢" if prob < 0.33 else "🟡" if prob < 0.66 else "🔴"

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", prediction)
    c2.metric("Chance", f"{prob*100:.1f}%")
    c3.metric("Risk Level", f"{color} {risk_level}")

    st.subheader("Why this result? (SHAP contributions to the hybrid probability)")
    sv = explainer(X_proc, max_evals=2 * X_proc.shape[1] + 1 + 200)
    contributions = pd.DataFrame({
        "Feature": hybrid.feature_names,
        "Value used by model": X_proc.iloc[0].values,
        "SHAP Impact": sv.values[0],
    })
    contributions["Effect"] = contributions["SHAP Impact"].apply(
        lambda v: "Raises risk ⬆" if v > 0.001 else ("Lowers risk ⬇" if v < -0.001 else "No real impact")
    )
    contributions = contributions.reindex(
        contributions["SHAP Impact"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    st.dataframe(contributions, hide_index=True, use_container_width=True)

    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(fig, use_container_width=True)

    st.info("⚠️ Educational/academic tool only — not a certified medical diagnosis.")
