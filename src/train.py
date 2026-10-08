"""
Model training and pipeline module for RetainIQ.
Trains:
1. Churn Classifiers: Logistic Regression baseline vs XGBoost (with scale_pos_weight for imbalance).
2. Survival Analysis: Kaplan-Meier Fitter by RFM segment + Cox Proportional Hazards Model.
3. LTV Regression: XGBoost Regressor for 90-day forward spend.
Saves all model artifacts, scalers, and pipelines with joblib.
"""

from pathlib import Path
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.features import load_or_engineer_features

MODELS_DIR = Path("models")
SEED = 42

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Tenure",
    "AvgOrderValue",
    "UniqueProducts",
    "IsUK",
]


def train_and_save_all_models():
    """Train classification, survival, and regression models and serialize them."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_engineer_features()
    print(f"[Train] Loaded dataset with {len(df):,} customers.")

    # ---------------------------------------------------------
    # 1. TRAIN / TEST SPLIT (Stratified on Churn)
    # ---------------------------------------------------------
    X = df[FEATURE_COLUMNS].copy()
    y_churn = df["Churn"].values
    y_ltv = df["FutureLTV"].values

    X_train, X_test, y_churn_train, y_churn_test, y_ltv_train, y_ltv_test, idx_train, idx_test = (
        train_test_split(
            X,
            y_churn,
            y_ltv,
            df.index,
            test_size=0.25,
            random_state=SEED,
            stratify=y_churn,
        )
    )

    print(f"[Train] Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # ---------------------------------------------------------
    # 2. CHURN CLASSIFIERS
    # ---------------------------------------------------------
    # Handle class imbalance: scale_pos_weight = (num_neg / num_pos)
    num_neg = (y_churn_train == 0).sum()
    num_pos = (y_churn_train == 1).sum()
    scale_pos_weight = num_neg / max(1, num_pos)

    # 2a. Logistic Regression Pipeline (with StandardScaler)
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")),
    ])
    print("[Train] Training Logistic Regression baseline...")
    lr_pipeline.fit(X_train, y_churn_train)

    # 2b. XGBoost Classifier Pipeline
    xgb_churn = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        eval_metric="logloss",
    )
    print("[Train] Training XGBoost Churn Classifier...")
    xgb_churn.fit(X_train, y_churn_train)

    # ---------------------------------------------------------
    # 3. LTV REGRESSION MODEL
    # ---------------------------------------------------------
    # Log-transform target for stability during training, invert during inference
    y_ltv_train_log = np.log1p(np.maximum(0, y_ltv_train))
    xgb_ltv = XGBRegressor(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
    )
    print("[Train] Training XGBoost LTV Regressor...")
    xgb_ltv.fit(X_train, y_ltv_train_log)

    # ---------------------------------------------------------
    # 4. SURVIVAL ANALYSIS (Lifelines)
    # ---------------------------------------------------------
    print("[Train] Fitting Cox Proportional Hazards Model...")
    cox_df = df.loc[idx_train, FEATURE_COLUMNS + ["SurvivalDuration", "SurvivalEvent"]].copy()
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_df, duration_col="SurvivalDuration", event_col="SurvivalEvent")

    # Fit Kaplan-Meier by segment on full training data
    kmf_dict = {}
    train_segments = df.loc[idx_train, "CustomerSegment"]
    train_durations = df.loc[idx_train, "SurvivalDuration"]
    train_events = df.loc[idx_train, "SurvivalEvent"]

    for segment in df["CustomerSegment"].unique():
        kmf = KaplanMeierFitter()
        mask = train_segments == segment
        if mask.sum() > 0:
            kmf.fit(train_durations[mask], event_observed=train_events[mask], label=segment)
            kmf_dict[segment] = kmf

    # Full population KMF
    kmf_global = KaplanMeierFitter()
    kmf_global.fit(train_durations, event_observed=train_events, label="Global Population")
    kmf_dict["Global"] = kmf_global

    # ---------------------------------------------------------
    # 5. SAVE ARTIFACTS
    # ---------------------------------------------------------
    joblib.dump(lr_pipeline, MODELS_DIR / "lr_churn_pipeline.joblib")
    joblib.dump(xgb_churn, MODELS_DIR / "xgb_churn_model.joblib")
    joblib.dump(xgb_ltv, MODELS_DIR / "xgb_ltv_model.joblib")
    joblib.dump(cph, MODELS_DIR / "cox_ph_model.joblib")
    joblib.dump(kmf_dict, MODELS_DIR / "kmf_segments.joblib")

    # Save test partition for evaluation
    test_data = {
        "X_test": X_test,
        "y_churn_test": y_churn_test,
        "y_ltv_test": y_ltv_test,
        "idx_test": idx_test,
        "FEATURE_COLUMNS": FEATURE_COLUMNS,
    }
    joblib.dump(test_data, MODELS_DIR / "test_data.joblib")

    print("[Train] All models and pipelines saved successfully to models/.")
    return test_data


if __name__ == "__main__":
    train_and_save_all_models()
