"""
data/generate_sample_data.py
===============================
Generates two synthetic, deliberately messy CSV datasets (one
classification-friendly, one regression-friendly) so AutoPilot-DS can be
demoed without network access to a real-world dataset source. Run:

    python data/generate_sample_data.py

Produces:
    data/sample_churn_messy.csv       (classification: churn)
    data/sample_house_prices_messy.csv (regression: price)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)


def make_churn_dataset(n: int = 1200) -> pd.DataFrame:
    tenure = RNG.integers(0, 72, n).astype(float)
    monthly_charges = np.round(RNG.normal(65, 30, n), 2)
    monthly_charges = np.clip(monthly_charges, 10, 200)
    contract = RNG.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = RNG.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
    payment = RNG.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    senior = RNG.choice([0, 1], n, p=[0.84, 0.16])
    customer_id = [f"CUST-{i:05d}" for i in range(n)]
    city = RNG.choice(["Chennai", "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune"], n)

    # churn probability driven by real signal (short tenure, month-to-month, high charges)
    logit = (
        -1.2
        + (-0.04 * tenure)
        + (0.015 * monthly_charges)
        + np.where(contract == "Month-to-month", 1.1, 0.0)
        + np.where(internet == "Fiber optic", 0.4, 0.0)
        + RNG.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (RNG.uniform(0, 1, n) < prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "customer_id": customer_id,
        "city": city,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "senior_citizen": senior,
        "churn": churn_label,
    })

    df = _inject_messiness(df, target_col="churn")
    return df


def make_house_price_dataset(n: int = 1000) -> pd.DataFrame:
    sqft = RNG.normal(1800, 650, n).clip(400, 6000)
    bedrooms = RNG.integers(1, 6, n)
    bathrooms = RNG.integers(1, 4, n)
    age_years = RNG.integers(0, 80, n)
    garage = RNG.choice([0, 1], n, p=[0.3, 0.7])
    neighborhood = RNG.choice(["Downtown", "Suburb", "Rural", "Waterfront"], n, p=[0.3, 0.4, 0.2, 0.1])
    listing_id = [f"L{i:06d}" for i in range(n)]

    base = (
        50000
        + sqft * 120
        + bedrooms * 8000
        - age_years * 350
        + garage * 12000
        + np.where(neighborhood == "Waterfront", 90000, 0)
        + np.where(neighborhood == "Downtown", 30000, 0)
    )
    price = np.round(base + RNG.normal(0, 25000, n), -2)
    price = np.clip(price, 40000, None)

    df = pd.DataFrame({
        "listing_id": listing_id,
        "sqft": np.round(sqft, 1),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "has_garage": garage,
        "neighborhood": neighborhood,
        "price": price,
    })
    df = _inject_messiness(df, target_col="price")
    return df


def _inject_messiness(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # missing values scattered across a few columns (never the target)
    for col in df.columns:
        if col == target_col:
            continue
        if RNG.uniform() < 0.6:
            frac = RNG.uniform(0.03, 0.18)
            idx = RNG.choice(n, size=int(n * frac), replace=False)
            df.loc[idx, col] = np.nan

    # duplicate rows
    dupe_idx = RNG.choice(n, size=max(1, int(n * 0.02)), replace=False)
    df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)

    # a constant column (data-quality noise)
    df["data_source_flag"] = "batch_import_2026"

    # inject some string-typed numbers (dtype messiness)
    numeric_like = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
    if numeric_like:
        col = numeric_like[0]
        idx = RNG.choice(len(df), size=max(1, int(len(df) * 0.05)), replace=False)
        df[col] = df[col].astype(object)
        df.loc[idx, col] = df.loc[idx, col].astype(str) + " "  # trailing space, still numeric-parseable

    # shuffle rows so it doesn't look synthetically ordered
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    churn_df = make_churn_dataset()
    house_df = make_house_price_dataset()
    churn_df.to_csv(os.path.join(out_dir, "sample_churn_messy.csv"), index=False)
    house_df.to_csv(os.path.join(out_dir, "sample_house_prices_messy.csv"), index=False)
    print(f"Wrote sample_churn_messy.csv {churn_df.shape}")
    print(f"Wrote sample_house_prices_messy.csv {house_df.shape}")
