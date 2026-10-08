"""
Evaluation and interpretability module for RetainIQ.
Evaluates:
- Churn Classification: F1, Precision, Recall, ROC-AUC, Confusion Matrix, ROC curves.
- Survival Analysis: Concordance index (C-index) for Cox PH.
- LTV Regression: MAE, RMSE, R-squared.
- SHAP global and local feature importance.
Outputs:
- models/metrics.json
- reports/figures/06_confusion_matrices.png
- reports/figures/07_roc_curves.png
- reports/figures/08_shap_summary.png
- reports/figures/09_kaplan_meier_survival.png
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)

MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")


def evaluate_all_models():
    """Compute and save comprehensive evaluation metrics and visualizations."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test_data = joblib.load(MODELS_DIR / "test_data.joblib")
    lr_pipeline = joblib.load(MODELS_DIR / "lr_churn_pipeline.joblib")
    xgb_churn = joblib.load(MODELS_DIR / "xgb_churn_model.joblib")
    xgb_ltv = joblib.load(MODELS_DIR / "xgb_ltv_model.joblib")
    cph = joblib.load(MODELS_DIR / "cox_ph_model.joblib")
    kmf_dict = joblib.load(MODELS_DIR / "kmf_segments.joblib")

    X_test = test_data["X_test"]
    y_churn_test = test_data["y_churn_test"]
    y_ltv_test = test_data["y_ltv_test"]
    idx_test = test_data["idx_test"]
    feature_cols = test_data["FEATURE_COLUMNS"]

    print(f"[Eval] Evaluating on test set ({len(X_test):,} samples)...")

    # ---------------------------------------------------------
    # 1. CLASSIFICATION EVALUATION
    # ---------------------------------------------------------
    # Predictions
    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]

    xgb_preds = xgb_churn.predict(X_test)
    xgb_probs = xgb_churn.predict_proba(X_test)[:, 1]

    metrics = {
        "churn_classification": {
            "logistic_regression": {
                "accuracy": round(float(accuracy_score(y_churn_test, lr_preds)), 4),
                "precision": round(float(precision_score(y_churn_test, lr_preds)), 4),
                "recall": round(float(recall_score(y_churn_test, lr_preds)), 4),
                "f1_score": round(float(f1_score(y_churn_test, lr_preds)), 4),
                "roc_auc": round(float(roc_auc_score(y_churn_test, lr_probs)), 4),
                "confusion_matrix": confusion_matrix(y_churn_test, lr_preds).tolist(),
            },
            "xgboost": {
                "accuracy": round(float(accuracy_score(y_churn_test, xgb_preds)), 4),
                "precision": round(float(precision_score(y_churn_test, xgb_preds)), 4),
                "recall": round(float(recall_score(y_churn_test, xgb_preds)), 4),
                "f1_score": round(float(f1_score(y_churn_test, xgb_preds)), 4),
                "roc_auc": round(float(roc_auc_score(y_churn_test, xgb_probs)), 4),
                "confusion_matrix": confusion_matrix(y_churn_test, xgb_preds).tolist(),
            },
        }
    }

    # ---------------------------------------------------------
    # 2. LTV REGRESSION EVALUATION
    # ---------------------------------------------------------
    # Invert log1p transform
    y_ltv_pred_log = xgb_ltv.predict(X_test)
    y_ltv_pred = np.expm1(np.maximum(0, y_ltv_pred_log))

    mae = float(mean_absolute_error(y_ltv_test, y_ltv_pred))
    rmse = float(root_mean_squared_error(y_ltv_test, y_ltv_pred))
    r2 = float(r2_score(y_ltv_test, y_ltv_pred))

    metrics["ltv_regression"] = {
        "xgboost_regressor": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(r2, 4),
            "mean_actual_ltv": round(float(np.mean(y_ltv_test)), 2),
            "mean_pred_ltv": round(float(np.mean(y_ltv_pred)), 2),
        }
    }

    # ---------------------------------------------------------
    # 3. SURVIVAL ANALYSIS EVALUATION
    # ---------------------------------------------------------
    c_index = float(cph.concordance_index_)
    metrics["survival_analysis"] = {
        "cox_proportional_hazards": {
            "concordance_index": round(c_index, 4),
            "summary_coefficients": {
                col: round(float(cph.params_[col]), 4) for col in feature_cols if col in cph.params_
            },
        }
    }

    # Save metrics.json
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Metrics saved to {MODELS_DIR / 'metrics.json'}")

    # ---------------------------------------------------------
    # 4. PLOTS & FIGURES
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    # 4a. Confusion Matrices Comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm_lr = confusion_matrix(y_churn_test, lr_preds)
    cm_xgb = confusion_matrix(y_churn_test, xgb_preds)

    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    axes[0].set_title(f"Logistic Regression Baseline\n(F1: {metrics['churn_classification']['logistic_regression']['f1_score']:.3f})", fontweight="bold")
    axes[0].set_ylabel("Actual Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Purples", cbar=False, ax=axes[1],
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    axes[1].set_title(f"XGBoost Classifier\n(F1: {metrics['churn_classification']['xgboost']['f1_score']:.3f})", fontweight="bold")
    axes[1].set_ylabel("Actual Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_confusion_matrices.png", dpi=300)
    plt.close()

    # 4b. ROC Curves Comparison
    fpr_lr, tpr_lr, _ = roc_curve(y_churn_test, lr_probs)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_churn_test, xgb_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {metrics['churn_classification']['logistic_regression']['roc_auc']:.3f})", color="#6B7C93", linestyle="--", linewidth=2)
    ax.plot(fpr_xgb, tpr_xgb, label=f"XGBoost Classifier (AUC = {metrics['churn_classification']['xgboost']['roc_auc']:.3f})", color="#635BFF", linewidth=2.5)
    ax.plot([0, 1], [0, 1], color="#E3E8EE", linestyle=":", linewidth=2, label="Random Guess")
    ax.set_title("ROC Curves: Baseline vs XGBoost Churn Model", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_roc_curves.png", dpi=300)
    plt.close()

    # 4c. SHAP Global Feature Importance
    print("[Eval] Calculating SHAP values for XGBoost Churn Classifier...")
    explainer = shap.TreeExplainer(xgb_churn)
    shap_values = explainer(X_test)

    # Save explainer for fast inference in app
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Global Feature Importance (XGBoost Churn)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4d. Kaplan-Meier Survival Curves by Segment
    fig, ax = plt.subplots(figsize=(9, 6))
    palette_colors = {
        "Champions": "#1FA971",
        "Loyal Customers": "#635BFF",
        "Potential Loyalists": "#00D4B2",
        "At Risk": "#F5A623",
        "Hibernating": "#E5484D",
        "Global": "#0A2540",
    }
    for label, kmf in kmf_dict.items():
        color = palette_colors.get(label, "#635BFF")
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2.2 if label != "Global" else 3, linestyle="--" if label == "Global" else "-")

    ax.set_title("Kaplan-Meier Survival Curves by Customer RFM Segment", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Timeline (Days after Cutoff)")
    ax.set_ylabel("Probability of Remaining Active")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_kaplan_meier_survival.png", dpi=300)
    plt.close()

    print("[Eval] All evaluation plots generated successfully in reports/figures/.")
    return metrics


if __name__ == "__main__":
    evaluate_all_models()
