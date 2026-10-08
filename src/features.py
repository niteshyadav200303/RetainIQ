"""
Feature engineering module for RetainIQ.
Implements strictly leakage-free time-based cutoff feature extraction:
- Historical window (BEFORE cutoff):
  * Recency: days since most recent purchase up to cutoff_date
  * Frequency: count of unique invoice dates / transactions before cutoff
  * Monetary: total spend before cutoff
  * Tenure: days since first purchase up to cutoff_date
  * AvgOrderValue: Monetary / Frequency
  * UniqueProducts: count of unique stock codes purchased before cutoff
  * Country: primary customer location (UK vs Non-UK)
  * RFM Segments: Champions, Loyal, Potential, At Risk, Hibernating
- Future observation window (90 days AFTER cutoff):
  * Churn label: 1 if customer made 0 purchases in [cutoff, cutoff + 90 days), else 0
  * Future LTV: total spend in [cutoff, cutoff + 90 days)
  * Days to event / Censoring for survival analysis
"""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
CLEANED_PARQUET_PATH = PROCESSED_DIR / "cleaned_transactions.parquet"
FEATURES_PARQUET_PATH = PROCESSED_DIR / "customer_features.parquet"
FEATURES_CSV_PATH = PROCESSED_DIR / "customer_features.csv"

# Project-wide Cutoff date: 90 days before final date in dataset
# Dataset ends: 2011-12-09. Cutoff: 2011-09-01 (100 days before end, giving exact 90-day window)
DEFAULT_CUTOFF_DATE = pd.Timestamp("2011-09-01 00:00:00")
PREDICTION_WINDOW_DAYS = 90


def build_rfm_segments(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Assign standard marketing RFM segment labels using quantiles:
    Champions, Loyal Customers, Potential Loyalists, At Risk, Hibernating.
    """
    df = df_features.copy()
    # Recency score (1 = longest time ago / bad, 4 = most recent / good)
    r_labels = [4, 3, 2, 1]
    # Frequency & Monetary score (1 = low, 4 = high)
    m_labels = [1, 2, 3, 4]

    # Handle duplicates in quantiles gracefully
    df["R_Score"] = pd.qcut(df["Recency"].rank(method="first"), q=4, labels=r_labels).astype(int)
    df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), q=4, labels=m_labels).astype(int)
    df["M_Score"] = pd.qcut(df["Monetary"].rank(method="first"), q=4, labels=m_labels).astype(int)
    df["RFM_Score"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)

    def assign_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        elif r >= 3 and (f >= 2 or m >= 2):
            return "Loyal Customers"
        elif r >= 3 and f == 1:
            return "Potential Loyalists"
        elif r <= 2 and (f >= 3 or m >= 3):
            return "At Risk"
        else:
            return "Hibernating"

    df["CustomerSegment"] = df.apply(assign_segment, axis=1)
    return df


def engineer_customer_features(
    df_tx: pd.DataFrame,
    cutoff_date: pd.Timestamp = DEFAULT_CUTOFF_DATE,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Extract strictly leak-free customer features prior to cutoff_date,
    and ground truth labels during the 90 days after cutoff_date.
    """
    df_tx["InvoiceDate"] = pd.to_datetime(df_tx["InvoiceDate"])
    future_end_date = cutoff_date + pd.Timedelta(days=prediction_window_days)

    print(f"[Features] Cutoff date: {cutoff_date.date()}")
    print(f"[Features] Future window: {cutoff_date.date()} to {future_end_date.date()} ({prediction_window_days} days)")

    # 1. Historical transactions strictly BEFORE cutoff
    df_history = df_tx[df_tx["InvoiceDate"] < cutoff_date].copy()
    print(f"[Features] Historical transactions: {len(df_history):,}")

    # 2. Future transactions strictly within [cutoff_date, future_end_date)
    df_future = df_tx[
        (df_tx["InvoiceDate"] >= cutoff_date) & (df_tx["InvoiceDate"] < future_end_date)
    ].copy()
    print(f"[Features] Future window transactions: {len(df_future):,}")

    # Aggregate features per customer from history
    # Group by CustomerID
    grouped = df_history.groupby("CustomerID")

    # Recency: days from last invoice to cutoff
    last_tx = grouped["InvoiceDate"].max()
    recency = (cutoff_date - last_tx).dt.total_seconds() / (24 * 3600)

    # First purchase / Tenure: days from first invoice to cutoff
    first_tx = grouped["InvoiceDate"].min()
    tenure = (cutoff_date - first_tx).dt.total_seconds() / (24 * 3600)

    # Frequency: count of unique invoice dates (distinct shopping trips)
    # Using invoice level count
    frequency = grouped["Invoice"].nunique()

    # Monetary: total spend
    monetary = grouped["TotalSpend"].sum()

    # Unique products
    unique_products = grouped["StockCode"].nunique()

    # Average Order Value
    avg_order_value = monetary / frequency

    # Primary Country (most common country for customer)
    country_series = grouped["Country"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "United Kingdom")
    is_uk = (country_series == "United Kingdom").astype(int)

    features_df = pd.DataFrame({
        "CustomerID": recency.index,
        "Recency": recency.values,
        "Frequency": frequency.values,
        "Monetary": monetary.values,
        "Tenure": tenure.values,
        "AvgOrderValue": avg_order_value.values,
        "UniqueProducts": unique_products.values,
        "Country": country_series.values,
        "IsUK": is_uk.values,
    })

    # Ground truth labels from future window
    future_grouped = df_future.groupby("CustomerID")
    future_spend = future_grouped["TotalSpend"].sum()
    future_orders = future_grouped["Invoice"].nunique()

    # Churn = 1 if customer made NO purchase in the 90-day window
    # Future LTV = total spend in those 90 days
    features_df["FutureSpend"] = features_df["CustomerID"].map(future_spend).fillna(0.0)
    features_df["FutureOrders"] = features_df["CustomerID"].map(future_orders).fillna(0).astype(int)
    features_df["Churn"] = (features_df["FutureOrders"] == 0).astype(int)
    features_df["FutureLTV"] = features_df["FutureSpend"]

    # Survival analysis duration and event indicators:
    # Duration = days from first transaction until either next transaction or censoring at study end
    # For customer-level survival during the 90 day window:
    # Event = Churned (1) or Retained (0)
    # Survival time (T): if returned, days until first return after cutoff; if churned, censored at 90 days
    first_future_tx = future_grouped["InvoiceDate"].min()
    features_df["FirstFutureTx"] = features_df["CustomerID"].map(first_future_tx)

    def calc_survival_duration(row):
        if pd.isna(row["FirstFutureTx"]):
            # Churned: did not return during 90-day window -> Censored at 90 days
            return float(prediction_window_days)
        else:
            # Retained: event occurred at (FirstFutureTx - cutoff)
            days = (row["FirstFutureTx"] - cutoff_date).total_seconds() / (24 * 3600)
            return max(1.0, min(float(prediction_window_days), days))

    features_df["SurvivalDuration"] = features_df.apply(calc_survival_duration, axis=1)
    # In survival analysis of retention: Event = 1 if returned (retention event) or Event = 1 if churned
    # We define Event = Churn (1 = Churned, 0 = Active/Retained)
    features_df["SurvivalEvent"] = features_df["Churn"]
    features_df.drop(columns=["FirstFutureTx"], inplace=True)

    # Segment customers
    features_df = build_rfm_segments(features_df)

    print(f"[Features] Created feature matrix for {len(features_df):,} customers.")
    churn_rate = features_df["Churn"].mean() * 100
    print(f"[Features] Overall 90-Day Churn Rate: {churn_rate:.2f}%")
    print(f"[Features] Avg 90-Day Future LTV: £{features_df['FutureLTV'].mean():.2f}")
    return features_df


def load_or_engineer_features(force_reload: bool = False) -> pd.DataFrame:
    """Load cached features or compute them from clean transactions."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not force_reload and FEATURES_PARQUET_PATH.exists():
        print(f"[Features] Loading cached features from {FEATURES_PARQUET_PATH}...")
        return pd.read_parquet(FEATURES_PARQUET_PATH)

    df_tx = pd.read_parquet(CLEANED_PARQUET_PATH)
    features_df = engineer_customer_features(df_tx)

    features_df.to_parquet(FEATURES_PARQUET_PATH, index=False)
    features_df.to_csv(FEATURES_CSV_PATH, index=False)
    print(f"[Features] Saved features to {FEATURES_PARQUET_PATH} and {FEATURES_CSV_PATH}")
    return features_df


if __name__ == "__main__":
    load_or_engineer_features(force_reload=True)
