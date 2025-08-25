import os
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------
# Helpers
# -----------------------

def ensure_dirs():
    """Ensure the necessary output directories exist."""
    Path("../models").mkdir(parents=True, exist_ok=True)
    Path("../outputs").mkdir(parents=True, exist_ok=True)

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a proper datetime column exists and is sorted."""
    if "date" not in df.columns:
        if "year" in df.columns and "month" in df.columns:
            df["year"] = df["year"].astype(int)
            df["month"] = df["month"].astype(int)
            df["date"] = pd.to_datetime(
                df["year"].astype(str) + "-" + df["month"].astype(str) + "-01",
                format="%Y-%m-%d"
            )
        else:
            raise ValueError("Input must contain 'date' or both 'year' and 'month'.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["account_id", "date"]).reset_index(drop=True)
    return df

def feature_engineering(df: pd.DataFrame, lags=(1,2,3,4,6,9,12)) -> pd.DataFrame:
    """Create lag and rolling features per account_id."""
    if "monthly_value" not in df.columns:
        raise ValueError("Expected 'monthly_value' column in dataset.")
    frames = []
    for acc, g in df.groupby("account_id"):
        g = g.copy().sort_values("date")
        
        g["monthly_value_safe"] = np.maximum(0, g["monthly_value"])
        g["monthly_value_log"] = np.log1p(g["monthly_value_safe"].values)
        
        g['pct_change_1'] = g['monthly_value_log'].pct_change().fillna(0.0)
        for L in lags:
            g[f"lag_{L}"] = g["monthly_value_log"].shift(L)
        g["roll_mean_3"] = g["monthly_value_log"].rolling(3).mean()
        g["roll_std_3"] = g["monthly_value_log"].rolling(3).std()
        g["roll_mean_6"] = g["monthly_value_log"].rolling(6).mean()
        g["roll_std_6"] = g["monthly_value_log"].rolling(6).std()
        g.fillna(method="ffill", inplace=True)
        g.fillna(0.0, inplace=True)
        frames.append(g)
    out = pd.concat(frames, axis=0).reset_index(drop=True)
    return out

def build_feature_matrix(df: pd.DataFrame, label_col="monthly_value_log"):
    """
    Builds the feature matrix and returns the LabelEncoders for saving.
    """
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["year"] = work["date"].dt.year.astype(int)
    work["month"] = work["date"].dt.month.astype(int)

    cat_encoders = {}
    for cat_col in ["dealer_code", "english_name", "account_id"]:
        if cat_col in work.columns:
            enc = LabelEncoder()
            work[cat_col + "_enc"] = enc.fit_transform(work[cat_col].astype(str))
            cat_encoders[cat_col] = enc
            
    feature_cols = [c for c in work.columns if c.startswith("lag_") or c.startswith("roll_")]
    feature_cols += ["year", "month", "pct_change_1"]
    if "dealer_code_enc" in work.columns: feature_cols.append("dealer_code_enc")
    if "english_name_enc" in work.columns: feature_cols.append("english_name_enc")
    if "account_id_enc" in work.columns: feature_cols.append("account_id_enc")

    X = work[feature_cols].values
    y = work[label_col].values
    
    return X, y, feature_cols, cat_encoders

# -----------------------
# Main pipeline
# -----------------------

def main(input_csv: str, output_models_dir="../models", output_dir="../outputs"):
    ensure_dirs()
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    for col in ["monthly_value", "yearly_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = add_time_columns(df)
    df_fe = feature_engineering(df)
    
    X_train, y_train, feat_cols, cat_encoders = build_feature_matrix(df_fe)

    if X_train.shape[0] < 5:
        print("Insufficient training samples after feature engineering.")
        return

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        min_data_in_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Calculate and print accuracy metrics
    train_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, train_pred)
    rmse = np.sqrt(mse)

    preprocessor = {
        'cat_encoders': cat_encoders,
        'feature_cols': feat_cols,
    }
    
    joblib.dump(model, os.path.join(output_models_dir, "lgbm_model.pkl"))
    joblib.dump(preprocessor, os.path.join(output_models_dir, "preprocessor.pkl"))

    print("\n=== Training complete (LightGBM, Global Model) ===")
    print(f"- Model saved to: {os.path.join(output_models_dir, 'lgbm_model.pkl')}")
    print(f"- Preprocessor saved to: {os.path.join(output_models_dir, 'preprocessor.pkl')}")
    print(f"\nTraining Root Mean Squared Error (RMSE): {rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Train a single global LightGBM model and preprocessor.")
    parser.add_argument("--input", type=str, default="../data/processed/cleaned_dataset.csv", help="Path to cleaned CSV (with year/month or date).")
    args = parser.parse_args()
    main(args.input)