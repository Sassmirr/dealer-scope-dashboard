"""
# forecast.py
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # points to project root (dealer/)
MODEL_PATH = BASE_DIR / "modelS" / "varmax_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "kpi_cleaned.csv"
FUTURE_PERIODS = 12  # Number of future time steps to forecast

# --- LOAD MODEL ---
with open(MODEL_PATH, "rb") as f:
    varmax_model = pickle.load(f)

# --- LOAD ORIGINAL DATA ---
df = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)

# --- FORECAST ---
forecast_result = varmax_model.get_forecast(steps=FUTURE_PERIODS)
forecast_df = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()

print("Forecasted Values:")
print(forecast_df)

# --- PLOT ---
plt.figure(figsize=(10, 5))
plt.plot(df.index, df, label="Actual")
plt.plot(
    pd.date_range(df.index[-1] + timedelta(1), periods=FUTURE_PERIODS, freq="MS"),
    forecast_df,
    label="Forecast",
    color="red",
)
plt.fill_between(
    pd.date_range(df.index[-1] + timedelta(1), periods=FUTURE_PERIODS, freq="MS"),
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink",
    alpha=0.3,
)
plt.title("VARMAX Forecast")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.tight_layout()
plt.show()
"""
# forecast.py
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import PROCESSED_FILE

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data/processed/kpi_cleaned.csv"
MODELS_DIR = BASE_DIR / "models"
FUTURE_PERIODS = 12

# --- Load processed data ---
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Processed data not found at {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH, parse_dates=['date'])
df_raw = df_raw.sort_values('date').reset_index(drop=True)
print(f"[INFO] Loaded data with columns: {df_raw.columns.tolist()}")

# --- Select model ---
MODEL_TYPE = "varmax"  # Change to "prophet" or "lgbm" as needed
MODEL_PATH = MODELS_DIR / f"{MODEL_TYPE}_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Saved model not found at {MODEL_PATH}")

# --- Forecast logic per model ---
if MODEL_TYPE == "varmax":
    from src.modelling.varmax import VARMAXModel

    with open(MODEL_PATH, "rb") as f:
        varmax_results = pickle.load(f)  # VARResults object

    # Prepare wide-format data (date x account_id)
    df_wide = df_raw.pivot(index="date", columns="account_id", values="monthly_value").fillna(0)

    # Forecast using last k_ar observations
    k_ar = varmax_results.k_ar
    last_obs = df_wide.values[-k_ar:]
    forecast_vals = varmax_results.forecast(y=last_obs, steps=FUTURE_PERIODS)

    # Convert to DataFrame with future dates
    freq = pd.infer_freq(df_wide.index) or "MS"
    future_dates = pd.date_range(df_wide.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                 periods=FUTURE_PERIODS, freq=freq)
    forecast_df = pd.DataFrame(forecast_vals, index=future_dates, columns=df_wide.columns)
    print("[INFO] VARMAX forecast complete.")

elif MODEL_TYPE == "prophet":
    from src.modelling.prophet_wrap import ProphetWrapper

    with open(MODEL_PATH, "rb") as f:
        prophet_model = pickle.load(f)

    future_df = prophet_model.make_future_dataframe(periods=FUTURE_PERIODS, freq='MS')
    forecast = prophet_model.predict(future_df)
    forecast_df = forecast[['ds', 'yhat']].set_index('ds')
    print("[INFO] Prophet forecast complete.")

elif MODEL_TYPE == "lgbm":
    from src.modelling.lgbm import LGBMModel

    with open(MODEL_PATH, "rb") as f:
        lgbm_model = pickle.load(f)

    feature_cols = [c for c in df_raw.columns if c.startswith('lag_') or 'rolling' in c]
    df_latest = df_raw.sort_values('date').iloc[-FUTURE_PERIODS:]
    X_future = df_latest[feature_cols].fillna(0)
    forecast_vals = lgbm_model.predict(X_future)

    freq = pd.infer_freq(df_raw['date']) or "MS"
    future_dates = pd.date_range(df_raw['date'].iloc[-1] + pd.tseries.frequencies.to_offset(freq),
                                 periods=FUTURE_PERIODS, freq=freq)
    forecast_df = pd.DataFrame(forecast_vals, index=future_dates, columns=['forecast'])
    print("[INFO] LGBM forecast complete.")

else:
    raise ValueError("Unknown MODEL_TYPE. Choose from: varmax / prophet / lgbm")

# --- Plot results ---
plt.figure(figsize=(12, 6))

# Historical data
if 'monthly_value' in df_raw.columns:
    plt.plot(df_raw['date'], df_raw['monthly_value'], label='Actual', color='blue')
else:
    plt.plot(df_raw['date'], df_raw.set_index('date'), label='Actual', color='blue')

# Forecast
plt.plot(forecast_df.index, forecast_df.iloc[:, 0], label='Forecast', color='red')

plt.title(f"{MODEL_TYPE.upper()} Forecast")
plt.xlabel("Date")
plt.ylabel("Monthly KPI Value")
plt.legend()
plt.tight_layout()
plt.show()
