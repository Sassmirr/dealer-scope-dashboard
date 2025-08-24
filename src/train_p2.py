# train_p2.py
# Phase 2: training + evaluation + saving models + 3-month forecast (LightGBM, accuracy optimized)

import os
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------
# Helpers
# -----------------------

def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    if "monthly_value" not in df.columns:
        raise ValueError("Expected 'monthly_value' column in dataset.")
    frames = []
    for acc, g in df.groupby("account_id"):
        g = g.copy().sort_values("date")
        # Log-transform target
        g["monthly_value_log"] = np.log1p(g["monthly_value"].values)
        # Percentage change
        g['pct_change_1'] = g['monthly_value_log'].pct_change().fillna(0.0)
        # Lags
        for L in lags:
            g[f"lag_{L}"] = g["monthly_value_log"].shift(L)
        # Rolling stats
        g["roll_mean_3"] = g["monthly_value_log"].rolling(3).mean()
        g["roll_std_3"]  = g["monthly_value_log"].rolling(3).std()
        g["roll_mean_6"] = g["monthly_value_log"].rolling(6).mean()
        g["roll_std_6"]  = g["monthly_value_log"].rolling(6).std()
        # Fill missing
        g.fillna(method="ffill", inplace=True)
        g.fillna(0.0, inplace=True)
        frames.append(g)
    out = pd.concat(frames, axis=0).reset_index(drop=True)
    return out

def time_split(group_df: pd.DataFrame, val_last_n_months: int = 3):
    group_df = group_df.sort_values("date")
    cutoff_idx = max(1, group_df.shape[0] - val_last_n_months)
    return group_df.iloc[:cutoff_idx].copy(), group_df.iloc[cutoff_idx:].copy()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if np.sum(mask) == 0: return np.nan
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)

def build_feature_matrix(df: pd.DataFrame, label_col="monthly_value_log"):
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["year"] = work["date"].dt.year.astype(int)
    work["month"] = work["date"].dt.month.astype(int)

    # encode categorical
    cat_encoders = {}
    for cat_col in ["dealer_code", "english_name"]:
        if cat_col in work.columns:
            enc = LabelEncoder()
            work[cat_col + "_enc"] = enc.fit_transform(work[cat_col].astype(str))
            cat_encoders[cat_col] = enc

    feature_cols = [c for c in work.columns if c.startswith("lag_") or c.startswith("roll_")]
    feature_cols += ["year","month","pct_change_1"]
    if "dealer_code_enc" in work.columns: feature_cols.append("dealer_code_enc")
    if "english_name_enc" in work.columns: feature_cols.append("english_name_enc")

    X = work[feature_cols].values
    y = work[label_col].values
    return X, y, feature_cols, cat_encoders

def train_per_account(group_df: pd.DataFrame, feature_cols=None):
    # Skip accounts with <12 months history
    if group_df.shape[0] < 12:
        return None, None, None

    train_df, valid_df = time_split(group_df)
    X_train, y_train, feat_cols, _ = build_feature_matrix(train_df)
    X_valid, y_valid, _, _ = build_feature_matrix(valid_df)

    # Check minimum samples
    if X_train.shape[0] < 2 or X_valid.shape[0] < 1:
        return None, None, None

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
    valid_pred = model.predict(X_valid)
    metrics = {
        "MAE": float(mean_absolute_error(y_valid, valid_pred)),
        "RMSE": rmse(y_valid, valid_pred),
        "SMAPE": smape(y_valid, valid_pred),
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid))
    }
    return model, feat_cols, metrics

def next_month(dt):
    dt = pd.to_datetime(dt, errors="coerce")
    y, m = dt.year, dt.month + 1
    if m == 13: y, m = y+1, 1
    return pd.Timestamp(year=y, month=m, day=1)

def recursive_forecast_3m(group_df, model, feature_cols):
    g = group_df.sort_values("date").copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    last_date = g["date"].max()
    working = g.copy()
    future_rows = []

    for h in range(1,4):
        fut_date = next_month(last_date) if h==1 else next_month(future_rows[-1]["date"])
        blank = {"account_id": g["account_id"].iloc[-1], "date": fut_date, "monthly_value_log": 0.0}
        tmp = pd.concat([working, pd.DataFrame([blank])], ignore_index=True)

        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp["year"] = tmp["date"].dt.year.astype(int)
        tmp["month"] = tmp["date"].dt.month.astype(int)

        tmp = feature_engineering(tmp)
        last_row = tmp.tail(1).copy()
        for fc in feature_cols:
            if fc not in last_row.columns: last_row[fc] = 0.0
        X_vec = last_row[feature_cols].values
        y_hat_log = float(model.predict(X_vec)[0])
        y_hat = np.expm1(y_hat_log)

        new_row = {"account_id": blank["account_id"], "date": fut_date, "monthly_value": y_hat}
        working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)
        future_rows.append({"account_id": blank["account_id"], "date": fut_date, "pred_monthly_value": y_hat})

    return pd.DataFrame(future_rows)

# -----------------------
# Main pipeline
# -----------------------

def main(input_csv: str, output_models_dir="models", output_dir="outputs"):
    ensure_dirs()
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    for col in ["monthly_value","yearly_value"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = add_time_columns(df)
    df_fe = feature_engineering(df)

    all_metrics, all_future, models_info = {}, [], []

    for acc_id, g in df_fe.groupby("account_id"):
        model, feat_cols, metrics = train_per_account(g)
        if model is None:  # skip small accounts
            continue
        all_metrics[str(acc_id)] = metrics

        model_path = Path(output_models_dir)/f"lgbm_account_{acc_id}.joblib"
        joblib.dump({"model": model, "feature_cols": feat_cols}, model_path)
        models_info.append({"account_id": acc_id, "model_path": str(model_path)})

        fut = recursive_forecast_3m(
            g[["account_id","date","monthly_value","dealer_code","english_name"]]
            if all(c in g.columns for c in ["dealer_code","english_name"])
            else g[["account_id","date","monthly_value"]],
            model, feat_cols
        )
        all_future.append(fut)

    # Save metrics and predictions
    with open(Path(output_dir)/"metrics.json","w") as f: json.dump(all_metrics,f,indent=2)

    future_df = pd.concat(all_future,ignore_index=True) if all_future else pd.DataFrame(columns=["account_id","date","pred_monthly_value"])
    hist = df[["account_id","date","monthly_value"]].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    future_df["date"] = pd.to_datetime(future_df["date"])
    out = hist.merge(future_df,on=["account_id","date"],how="outer").sort_values(["account_id","date"]).reset_index(drop=True)
    out["year"], out["month"] = out["date"].dt.year, out["date"].dt.month

    out_path = Path(output_dir)/"predictions.csv"
    out.to_csv(out_path,index=False)

    # --- Print overall accuracy ---
    all_mae = [m["MAE"] for m in all_metrics.values()]
    all_rmse = [m["RMSE"] for m in all_metrics.values()]
    all_smape = [m["SMAPE"] for m in all_metrics.values() if not np.isnan(m["SMAPE"])]
    print("\n=== Training complete (LightGBM Phase 2, Accuracy Optimized) ===")
    print(f"- Models saved under: {Path(output_models_dir).resolve()}")
    print(f"- Metrics saved to:   {Path(output_dir,'metrics.json').resolve()}")
    print(f"- Predictions saved:  {out_path.resolve()}")
    print("\n--- Overall Metrics ---")
    print(f"Average MAE:   {np.mean(all_mae):.4f}")
    print(f"Average RMSE:  {np.mean(all_rmse):.4f}")
    print(f"Average SMAPE: {np.mean(all_smape):.2f}%")
    conclusion = "Good" if np.mean(all_smape)<20 else "Average" if np.mean(all_smape)<35 else "Poor"
    print(f"Overall Model Quality: {conclusion}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Train LightGBM model(s) and forecast next 3 months.")
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned CSV (with year/month or date).")
    args = parser.parse_args()
    main(args.input)
