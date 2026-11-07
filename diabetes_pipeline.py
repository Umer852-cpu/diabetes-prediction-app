"""
diabetes_pipeline.py  (v4 - leakage-free version)

Explainable Diabetes Risk Prediction - Hybrid Machine Learning System
Muhammad Umer Hanif | 303-241031 | BS AI, Hazara University Mansehra

What changed from v3 (and why):
  * DATA LEAKAGE FIX
      - The data is split into train/test FIRST. All cleaning statistics
        (medians for missing values, outlier limits) are learned from the
        TRAINING set only and then applied to the test set / new patients.
      - Missing values are filled with the overall training median.
        v3 used the median of patients with the same Outcome, which leaks
        the answer (the label) into the Insulin / SkinThickness columns.
  * HYBRID WEIGHTS FIX
      - v3 computed the weights from TEST-set accuracy (test data used for
        model building) and the weights were almost equal (0.23-0.26), so the
        hybrid behaved like a plain average and tied with XGBoost.
      - v4 learns the weights from 5-fold OUT-OF-FOLD predictions on the
        TRAINING set only, by minimising log-loss (non-negative, sum to 1).
  * HONEST EVALUATION
      - Results are reported on the untouched test set AND with 5x5 repeated
        stratified cross-validation (whole pipeline re-run inside every fold),
        with a corrected resampled t-test (Nadeau & Bengio) for Hybrid vs XGBoost.
      - Baselines added: equal-weight soft voting and stacking.
      - Ablation: with vs without the engineered features.
  * SHAP now explains the HYBRID model itself (model-agnostic explainer),
    globally and for a single patient.

Run:  python diabetes_pipeline.py
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from scipy.optimize import minimize

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, log_loss
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SEED = 42
COLUMNS_WITH_HIDDEN_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ENGINEERED = ["Glucose_BMI_Score", "Insulin_Efficiency", "Older_Than_30", "High_BMI"]
BASE_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


# ======================================================================
# PREPROCESSING - learned from training data only (this is the leak fix)
# ======================================================================
class Preprocessor:
    """Fit on TRAINING rows only; then transform train, test and new patients."""

    def __init__(self, add_features=True):
        self.add_features = add_features
        self.medians = {}
        self.clip_limits = {}

    def fit(self, X):
        X = X.copy()
        for col in COLUMNS_WITH_HIDDEN_MISSING:
            X[col] = X[col].replace(0, np.nan)
            self.medians[col] = float(X[col].median())       # overall median, NO Outcome used
        # outlier limits from REAL (observed) values only - computing them after imputation
        # would squeeze the IQR (49% of Insulin is imputed) and clip valid measurements
        for col in X.columns:
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)   # NaNs are skipped
            iqr = q3 - q1
            self.clip_limits[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        return self

    def transform(self, X):
        X = X.copy()
        for col in COLUMNS_WITH_HIDDEN_MISSING:
            X[col] = X[col].replace(0, np.nan).fillna(self.medians[col])
        for col, (lo, hi) in self.clip_limits.items():
            X[col] = X[col].clip(lo, hi)
        if self.add_features:
            X = add_engineered_features(X)
        return X

    def to_dict(self):
        return {"medians": self.medians, "clip_limits": self.clip_limits,
                "add_features": self.add_features}

    @classmethod
    def from_dict(cls, d):
        p = cls(d["add_features"])
        p.medians = d["medians"]
        p.clip_limits = {k: tuple(v) for k, v in d["clip_limits"].items()}
        return p


def add_engineered_features(X):
    X = X.copy()
    X["Glucose_BMI_Score"] = (X["Glucose"] * X["BMI"]) / 100           # glucose-BMI interaction
    X["Insulin_Efficiency"] = X["Insulin"] / (X["Glucose"] + 1)        # insulin relative to glucose
    X["Older_Than_30"] = (X["Age"] > 30).astype(int)                   # simple age risk flag
    X["High_BMI"] = (X["BMI"] >= 30).astype(int)                       # clinical obesity flag
    return X


def make_base_models(seed=SEED):
    """Hyper-parameters fixed in advance (not tuned on the test set)."""
    return {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000, random_state=seed)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                 eval_metric="logloss", random_state=seed),
    }


# ======================================================================
# HYBRID MODEL - weights learned from out-of-fold predictions (weight fix)
# ======================================================================
class WeightedHybrid:
    """
    Weighted soft-voting hybrid:  P(x) = sum_i w_i * p_i(x),  w_i >= 0,  sum w_i = 1.
    The weights are chosen to minimise log-loss on 5-fold OUT-OF-FOLD predictions
    of the TRAINING set, so the test set is never used to build the model.
    """

    def __init__(self, add_features=True, seed=SEED):
        self.add_features = add_features
        self.seed = seed

    def _oof_probabilities(self, X_raw, y):
        P = np.zeros((len(X_raw), len(BASE_NAMES)))
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        for tr, va in folds.split(X_raw, y):
            prep = Preprocessor(self.add_features).fit(X_raw.iloc[tr])
            A, B = prep.transform(X_raw.iloc[tr]), prep.transform(X_raw.iloc[va])
            for j, model in enumerate(make_base_models(self.seed).values()):
                model.fit(A, y.iloc[tr])
                P[va, j] = model.predict_proba(B)[:, 1]
        return P

    def fit(self, X_raw, y):
        # 1) learn the weights from out-of-fold predictions on training data
        P = self._oof_probabilities(X_raw, y)
        self.oof_probs_ = P
        objective = lambda w: log_loss(y, np.clip(P @ w, 1e-6, 1 - 1e-6))
        n = len(BASE_NAMES)
        result = minimize(objective, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        w = np.clip(result.x, 0, None)
        self.weights = dict(zip(BASE_NAMES, w / w.sum()))

        # 2) refit preprocessing + all base models on the full training set
        self.prep = Preprocessor(self.add_features).fit(X_raw)
        Xp = self.prep.transform(X_raw)
        self.feature_names = list(Xp.columns)
        self.models = make_base_models(self.seed)
        for model in self.models.values():
            model.fit(Xp, y)
        return self

    def base_probabilities(self, X_raw):
        Xp = self.prep.transform(X_raw)
        return {n: m.predict_proba(Xp)[:, 1] for n, m in self.models.items()}

    def predict_proba_processed(self, Xp):
        """Hybrid probability from already-preprocessed features (used by SHAP)."""
        Xp = pd.DataFrame(Xp, columns=self.feature_names)
        return sum(self.weights[n] * m.predict_proba(Xp)[:, 1] for n, m in self.models.items())

    def predict_proba(self, X_raw):
        return self.predict_proba_processed(self.prep.transform(X_raw))


def stacking_probabilities(hybrid, y_train, X_test_raw):
    """Stacking baseline: logistic-regression meta-learner on the same OOF predictions."""
    meta = LogisticRegression(max_iter=1000).fit(hybrid.oof_probs_, y_train)
    P_test = np.column_stack(list(hybrid.base_probabilities(X_test_raw).values()))
    return meta.predict_proba(P_test)[:, 1]


def scores(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred),
        "F1-Score": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, prob),
    }


def evaluate_split(X_train, X_test, y_train, y_test, add_features=True):
    """Fit everything on the training part, score everything on the test part."""
    hybrid = WeightedHybrid(add_features).fit(X_train, y_train)
    probs = hybrid.base_probabilities(X_test)
    probs["Equal-Weight Voting"] = np.mean([probs[n] for n in BASE_NAMES], axis=0)
    probs["Stacking (LR meta)"] = stacking_probabilities(hybrid, y_train, X_test)
    probs["Hybrid Model"] = hybrid.predict_proba(X_test)
    return {name: scores(y_test, p) for name, p in probs.items()}, probs, hybrid


def corrected_resampled_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected t-test for repeated k-fold CV scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var) if var > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


# ======================================================================
# MAIN PIPELINE
# ======================================================================
class DiabetesRiskSystem:

    def __init__(self, data_path, output_dir="outputs"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.plot_dir = os.path.join(output_dir, "plots")
        self.model_dir = os.path.join(output_dir, "models")
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    # STEP 1: load raw data and split FIRST -----------------------------
    def load_and_split(self, test_size=0.2):
        self.raw = pd.read_csv(self.data_path)
        X = self.raw.drop(columns="Outcome")
        y = self.raw["Outcome"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=SEED)
        zeros = {c: int((X[c] == 0).sum()) for c in COLUMNS_WITH_HIDDEN_MISSING}
        print(f"Loaded {len(X)} patients | train {len(self.X_train)} | test {len(self.X_test)}")
        print(f"Hidden missing values (zeros): {zeros}")
        return self

    # STEP 2: EDA plots (class balance on raw data, correlations on TRAINING data)
    def save_eda_plots(self):
        plt.figure(figsize=(5, 4))
        self.raw["Outcome"].value_counts().sort_index().plot(kind="bar", color=["#2c3e50", "#c0392b"])
        plt.title("Class balance (768 patients)")
        plt.xticks([0, 1], ["No Diabetes (500)", "Diabetes (268)"], rotation=0)
        plt.ylabel("Number of patients")
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/class_balance.png", dpi=150)
        plt.close()

        prep = Preprocessor().fit(self.X_train)
        train_df = prep.transform(self.X_train).assign(Outcome=self.y_train.values)
        plt.figure(figsize=(9, 7))
        sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    annot_kws={"size": 7})
        plt.title("Feature correlations (training set, after cleaning)")
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/correlation_heatmap.png", dpi=150)
        plt.close()
        print("EDA plots saved.")
        return self

    # STEP 3: train on training set, evaluate once on the held-out test set
    def train_and_evaluate_holdout(self):
        self.test_metrics, self.test_probs, self.hybrid = evaluate_split(
            self.X_train, self.X_test, self.y_train, self.y_test)

        print("\nHybrid weights (learned from out-of-fold training predictions):")
        for n, w in self.hybrid.weights.items():
            print(f"  {n}: {w:.3f}")

        df = pd.DataFrame(self.test_metrics).T.round(4)
        df.to_csv(f"{self.output_dir}/model_results_test_set.csv")
        print("\n=== Held-out test set (154 patients) ===")
        print(df)

        # save fitted models + preprocessing so predict.py / app.py reuse them exactly
        for n, m in self.hybrid.models.items():
            joblib.dump(m, f"{self.model_dir}/{n.replace(' ', '_').lower()}.joblib")
        config = {"weights": self.hybrid.weights,
                  "feature_names": self.hybrid.feature_names,
                  "preprocessing": self.hybrid.prep.to_dict(),
                  "threshold": 0.5}
        with open(f"{self.model_dir}/hybrid_config.json", "w") as f:
            json.dump(config, f, indent=2)
        # small background sample (training data only) for SHAP explanations
        bg = self.hybrid.prep.transform(self.X_train).sample(100, random_state=SEED)
        bg.to_csv(f"{self.model_dir}/shap_background.csv", index=False)
        print("Models, preprocessing config and SHAP background saved.")
        return self

    # STEP 4: repeated cross-validation (whole pipeline inside each fold)
    def cross_validate(self, n_splits=5, n_repeats=5):
        X = self.raw.drop(columns="Outcome")
        y = self.raw["Outcome"]
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
        rows, ablation = [], []
        for fold, (tr, te) in enumerate(rskf.split(X, y)):
            m, _, h = evaluate_split(X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], add_features=True)
            for name, s in m.items():
                rows.append({"fold": fold, "Model": name, **s})
            m_raw, _, _ = evaluate_split(X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], add_features=False)
            ablation.append({"fold": fold,
                             "Hybrid (12 features) Acc": m["Hybrid Model"]["Accuracy"],
                             "Hybrid (8 raw features) Acc": m_raw["Hybrid Model"]["Accuracy"],
                             "Hybrid (12 features) AUC": m["Hybrid Model"]["ROC-AUC"],
                             "Hybrid (8 raw features) AUC": m_raw["Hybrid Model"]["ROC-AUC"]})
            print(f"  CV fold {fold + 1}/{n_splits * n_repeats} done")

        self.cv_folds = pd.DataFrame(rows)
        self.cv_folds.to_csv(f"{self.output_dir}/cv_fold_scores.csv", index=False)
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        mean = self.cv_folds.groupby("Model")[metrics].mean()
        std = self.cv_folds.groupby("Model")[metrics].std()
        summary = mean.round(4).astype(str) + " ± " + std.round(4).astype(str)
        order = mean.sort_values("ROC-AUC", ascending=False).index
        self.cv_mean, self.cv_std = mean.loc[order], std.loc[order]
        summary.loc[order].to_csv(f"{self.output_dir}/model_results_cv.csv")
        self.cv_mean.round(4).to_csv(f"{self.output_dir}/model_results.csv")
        print(f"\n=== {n_repeats}x{n_splits}-fold cross-validation (mean ± std) ===")
        print(summary.loc[order].to_string())

        # significance tests: Hybrid vs every other model
        n_test = len(X) // n_splits
        n_train = len(X) - n_test
        tests = []
        hyb = self.cv_folds[self.cv_folds.Model == "Hybrid Model"].sort_values("fold")
        for other in [m for m in order if m != "Hybrid Model"]:
            o = self.cv_folds[self.cv_folds.Model == other].sort_values("fold")
            for metric in ["Accuracy", "ROC-AUC"]:
                t, p = corrected_resampled_ttest(hyb[metric].values, o[metric].values, n_train, n_test)
                tests.append({"Comparison": f"Hybrid vs {other}", "Metric": metric,
                              "Mean difference": round(hyb[metric].mean() - o[metric].mean(), 4),
                              "t": round(t, 3), "p-value": round(p, 4)})
        self.tests = pd.DataFrame(tests)
        self.tests.to_csv(f"{self.output_dir}/significance_tests.csv", index=False)
        print("\n=== Corrected resampled t-tests (Nadeau & Bengio) ===")
        print(self.tests.to_string(index=False))

        abl = pd.DataFrame(ablation)
        abl_summary = abl.drop(columns="fold").agg(["mean", "std"]).T.round(4)
        t, p = corrected_resampled_ttest(abl["Hybrid (12 features) AUC"], abl["Hybrid (8 raw features) AUC"],
                                         n_train, n_test)
        abl_summary.loc["AUC difference p-value"] = [round(p, 4), np.nan]
        abl_summary.to_csv(f"{self.output_dir}/ablation_engineered_features.csv")
        print("\n=== Ablation: engineered features ===")
        print(abl_summary)
        return self

    # STEP 5: plots --------------------------------------------------------
    def save_result_plots(self):
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        short = {"Logistic Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF",
                 "Equal-Weight Voting": "Equal\nvoting", "Stacking (LR meta)": "Stacking",
                 "Hybrid Model": "Hybrid"}
        plot_df = self.cv_mean[metrics].rename(index=short)
        ax = plot_df.plot(kind="bar", yerr=self.cv_std[metrics].rename(index=short).values.T,
                          figsize=(7, 5), colormap="viridis", capsize=1.5, width=0.82, fontsize=12)
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Score (mean ± SD, 25 CV folds)", fontsize=12)
        ax.set_xlabel("")
        ax.legend(loc="upper center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.0))
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/model_comparison.png", dpi=150)
        plt.close()

        plt.figure(figsize=(5.2, 4.8))
        for name, prob in self.test_probs.items():
            if name in ("Equal-Weight Voting", "Stacking (LR meta)"):
                continue
            fpr, tpr, _ = roc_curve(self.y_test, prob)
            auc = roc_auc_score(self.y_test, prob)
            plt.plot(fpr, tpr, label=f"{name} ({auc:.3f})",
                     linewidth=3 if name == "Hybrid Model" else 1.4)
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.legend(fontsize=10, loc="lower right", title="Model (AUC)", title_fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/roc_curves.png", dpi=150)
        plt.close()

        hybrid_pred = (self.test_probs["Hybrid Model"] >= 0.5).astype(int)
        cm = confusion_matrix(self.y_test, hybrid_pred)
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
        plt.title("Hybrid Model - Confusion Matrix (test set)")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/confusion_matrix.png", dpi=150)
        plt.close()

        names = list(self.test_probs.keys())
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        rows = []
        for ax, name in zip(axes.flatten(), names):
            pred = (self.test_probs[name] >= 0.5).astype(int)
            cm = confusion_matrix(self.y_test, pred)
            tn, fp, fn, tp = cm.ravel()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
            ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            rows.append({"Model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                         "Sensitivity (TP rate)": round(tp / (tp + fn), 4),
                         "Specificity (TN rate)": round(tn / (tn + fp), 4)})
        for ax in axes.flatten()[len(names):]:
            ax.axis("off")
        plt.suptitle("Confusion matrices - all models (held-out test set)", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/confusion_matrices_all_models.png", dpi=150)
        plt.close()
        pd.DataFrame(rows).to_csv(f"{self.output_dir}/confusion_matrix_counts.csv", index=False)
        print("Result plots saved.")
        return self

    # STEP 6: SHAP on the HYBRID model itself -------------------------------
    def explain_with_shap(self):
        h = self.hybrid
        background = h.prep.transform(self.X_train).sample(100, random_state=SEED)
        X_test_p = h.prep.transform(self.X_test)
        explainer = shap.PermutationExplainer(h.predict_proba_processed, background,
                                              seed=SEED)
        sv = explainer(X_test_p, max_evals=2 * X_test_p.shape[1] + 1 + 200)
        sv.feature_names = h.feature_names

        plt.figure()
        shap.summary_plot(sv.values, X_test_p, show=False)
        plt.xlabel("SHAP value (impact on hybrid diabetes probability)")
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(sv.values, X_test_p, plot_type="bar", show=False)
        plt.xlabel("mean |SHAP value| (impact on hybrid probability)")
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

        importance = pd.Series(np.abs(sv.values).mean(axis=0), index=h.feature_names) \
            .sort_values(ascending=False)
        importance.to_csv(f"{self.output_dir}/feature_importance.csv", header=["mean_abs_shap_value"])
        print("\nHybrid-model SHAP importance:")
        print(importance.round(4))

        # local (per-patient) explanation: the test patient with the highest hybrid risk
        probs = self.test_probs["Hybrid Model"]
        idx = int(np.argmax(probs))
        self.local_patient = {"test_index": int(self.X_test.index[idx]), "probability": float(probs[idx]),
                              "actual_outcome": int(self.y_test.iloc[idx])}
        plt.figure()
        shap.plots.waterfall(sv[idx], max_display=10, show=False)
        plt.title(f"Local explanation - one test patient (hybrid risk = {probs[idx]:.2f})", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/shap_local_patient.png", dpi=150, bbox_inches="tight")
        plt.close()
        with open(f"{self.output_dir}/shap_local_patient.json", "w") as f:
            json.dump(self.local_patient, f, indent=2)
        print(f"Local explanation saved for test patient {self.local_patient}")
        return self


if __name__ == "__main__":
    system = DiabetesRiskSystem(data_path="data/diabetes_raw.csv")
    (
        system
        .load_and_split()
        .save_eda_plots()
        .train_and_evaluate_holdout()
        .cross_validate()
        .save_result_plots()
        .explain_with_shap()
    )
    print("\nAll done. Check the outputs/ folder for plots, models, and results.")
