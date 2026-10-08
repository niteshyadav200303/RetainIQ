"""
Data acquisition and preprocessing pipeline for RetainIQ.
Loads raw UCI Online Retail II sheets, cleans data, and saves standard processed data.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EXCEL_PATH = RAW_DIR / "online_retail_II.xlsx"
CLEANED_CSV_PATH = PROCESSED_DIR / "cleaned_transactions.csv"
CLEANED_PARQUET_PATH = PROCESSED_DIR / "cleaned_transactions.parquet"


def download_and_extract_raw_data() -> Path:
    """Download and extract raw dataset if not present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if EXCEL_PATH.exists():
        print(f"[Data] Found {EXCEL_PATH}")
        return EXCEL_PATH

    zip_file = RAW_DIR / "online_retail_ii.zip"
    if not zip_file.exists():
        print(f"[Data] Downloading from {DATA_URL}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(DATA_URL, headers=headers)
        with urllib.request.urlopen(req) as resp, open(zip_file, "wb") as out_f:
            out_f.write(resp.read())

    print("[Data] Extracting zip archive...")
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(RAW_DIR)
    return EXCEL_PATH


def load_raw_transactions() -> pd.DataFrame:
    """Load both sheets from Excel file into a single DataFrame."""
    if not EXCEL_PATH.exists():
        download_and_extract_raw_data()

    print("[Data] Loading sheet 'Year 2009-2010'...")
    df_2009 = pd.read_excel(EXCEL_PATH, sheet_name="Year 2009-2010")
    print(f"       Loaded {len(df_2009):,} rows.")

    print("[Data] Loading sheet 'Year 2010-2011'...")
    df_2010 = pd.read_excel(EXCEL_PATH, sheet_name="Year 2010-2011")
    print(f"       Loaded {len(df_2010):,} rows.")

    # Standardize column names if needed
    col_mapping = {
        "Invoice": "Invoice",
        "StockCode": "StockCode",
        "Description": "Description",
        "Quantity": "Quantity",
        "InvoiceDate": "InvoiceDate",
        "Price": "Price",
        "Customer ID": "CustomerID",
        "CustomerID": "CustomerID",
        "Country": "Country",
    }
    df_2009 = df_2009.rename(columns=col_mapping)
    df_2010 = df_2010.rename(columns=col_mapping)

    df_combined = pd.concat([df_2009, df_2010], ignore_index=True)
    print(f"[Data] Total combined raw records: {len(df_combined):,}")
    return df_combined


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean retail transactions according to strict rules:
    1. Drop null CustomerID (guest purchases cannot be tracked for churn/LTV).
    2. Drop cancellations (Invoice starts with 'C').
    3. Drop Quantity <= 0 and Price <= 0.
    4. Drop duplicates.
    5. Calculate TotalSpend = Quantity * Price.
    6. Convert InvoiceDate to datetime and CustomerID to integer string.
    """
    initial_len = len(df)

    # 1. Null CustomerID
    df_clean = df.dropna(subset=["CustomerID"]).copy()
    null_cust_dropped = initial_len - len(df_clean)

    # Convert CustomerID to integer formatted string/int
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)

    # 2. Cancellations (Invoice string starting with 'C')
    df_clean["Invoice"] = df_clean["Invoice"].astype(str)
    no_cancel = ~df_clean["Invoice"].str.startswith("C", na=False)
    cancel_dropped = (~no_cancel).sum()
    df_clean = df_clean[no_cancel].copy()

    # 3. Bad quantity or price
    valid_quant_price = (df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)
    bad_qp_dropped = (~valid_quant_price).sum()
    df_clean = df_clean[valid_quant_price].copy()

    # 4. Duplicates
    dup_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates().copy()

    # 5. TotalSpend & Datetime formatting
    df_clean["TotalSpend"] = df_clean["Quantity"] * df_clean["Price"]
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

    # 6. Clean Country whitespace and cast text columns to string
    df_clean["Country"] = df_clean["Country"].astype(str).str.strip()
    df_clean["StockCode"] = df_clean["StockCode"].astype(str)
    df_clean["Description"] = df_clean["Description"].fillna("").astype(str)

    print(f"[Clean] Initial records:        {initial_len:,}")
    print(f"[Clean] Dropped Null Customers: {null_cust_dropped:,}")
    print(f"[Clean] Dropped Cancellations:  {cancel_dropped:,}")
    print(f"[Clean] Dropped <=0 Qty/Price:  {bad_qp_dropped:,}")
    print(f"[Clean] Dropped Duplicates:     {dup_count:,}")
    print(f"[Clean] Final clean records:    {len(df_clean):,}")
    print(f"[Clean] Unique Customers:       {df_clean['CustomerID'].nunique():,}")
    print(f"[Clean] Date Range:             {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")

    return df_clean


def load_or_process_clean_data(force_reload: bool = False) -> pd.DataFrame:
    """Load cached cleaned data or run the full pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not force_reload and CLEANED_PARQUET_PATH.exists():
        print(f"[Data] Loading cached cleaned data from {CLEANED_PARQUET_PATH}...")
        return pd.read_parquet(CLEANED_PARQUET_PATH)

    df_raw = load_raw_transactions()
    df_clean = clean_transactions(df_raw)

    print(f"[Data] Saving clean data to {CLEANED_PARQUET_PATH} and {CLEANED_CSV_PATH}...")
    df_clean.to_parquet(CLEANED_PARQUET_PATH, index=False)
    # Also save CSV for portability
    df_clean.to_csv(CLEANED_CSV_PATH, index=False)
    return df_clean


if __name__ == "__main__":
    load_or_process_clean_data()
