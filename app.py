import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# =========================
# Page / App Config
# =========================
st.set_page_config(page_title="DealerScope AI – Predictive Analytics Dashboard", page_icon="🚗", layout="wide")
st.title("DealerScope AI – Predictive Analytics Dashboard 🚗")
st.markdown("This dashboard predicts KPI values for the next 3 months, shows correlations, and lets you run simple 'what-if' scenarios.")

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_DIR, "data/processed/cleaned_dataset.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "src/models/lgbm_model.pkl")
PREPROC_PATH = os.path.join(SCRIPT_DIR, "src/models/preprocessor.pkl")
SARIMAX_FORECAST_PATH = os.path.join(SCRIPT_DIR, "src/outputs/sarimax_forecast2.csv")

# =========================
# Helpers: Feature Engineering & Encoding
# =========================
def feature_engineering(df: pd.DataFrame, lags=(1, 2, 3, 4, 6, 9, 12)) -> pd.DataFrame:
    if "monthly_value" not in df.columns:
        raise ValueError("Expected 'monthly_value' column in dataset.")
    frames = []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for _, g in df.groupby("account_id", sort=False):
        g = g.copy().sort_values("date")
        g["monthly_value_safe"] = np.maximum(0, g["monthly_value"].values)
        g["monthly_value_log"] = np.log1p(g["monthly_value_safe"].values)
        g["pct_change_1"] = g["monthly_value_log"].pct_change().fillna(0.0)
        for L in lags:
            g[f"lag_{L}"] = g["monthly_value_log"].shift(L)
        g["roll_mean_3"] = g["monthly_value_log"].rolling(3).mean()
        g["roll_std_3"] = g["monthly_value_log"].rolling(3).std()
        g["roll_mean_6"] = g["monthly_value_log"].rolling(6).mean()
        g["roll_std_6"] = g["monthly_value_log"].rolling(6).std()
        g.fillna(method="ffill", inplace=True)
        g.fillna(0.0, inplace=True)
        frames.append(g)
    return pd.concat(frames, axis=0).reset_index(drop=True)

def create_features_and_encode(df: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    work_df = feature_engineering(df.copy())
    if preprocessor and isinstance(preprocessor, dict) and "cat_encoders" in preprocessor:
        for cat_col, enc in preprocessor["cat_encoders"].items():
            if cat_col in work_df.columns:
                try:
                    work_df[cat_col + "_enc"] = enc.transform(work_df[cat_col].astype(str))
                except Exception:
                    work_df[cat_col + "_enc"] = -1
    return work_df

# =========================
# Data / Model Loaders
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame | None:
    if not os.path.exists(path):
        st.error(f"Dataset not found at: {os.path.abspath(path)}")
        return None
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns and {"year", "month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
        else:
            df["date"] = pd.to_datetime(df["date"])
        req = {"account_id", "english_name", "dealer_code", "date", "monthly_value"}
        missing = req - set(df.columns)
        if missing:
            st.error(f"Dataset is missing required columns: {sorted(missing)}")
            return None
        return df.sort_values(["account_id", "date"]).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error while loading the dataset: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_model_and_preprocessor(model_path: str = MODEL_PATH, preproc_path: str = PREPROC_PATH):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {os.path.abspath(model_path)}")
        return None, None
    if not os.path.exists(preproc_path):
        st.error(f"Preprocessor file not found at: {os.path.abspath(preproc_path)}")
        return None, None
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preproc_path)
        if not isinstance(preprocessor, dict) or "feature_cols" not in preprocessor:
            st.error("Preprocessor object must be a dict with a 'feature_cols' key.")
            return None, None
        return model, preprocessor
    except Exception as e:
        st.error(f"Error while loading model/preprocessor: {e}")
        return None, None

@st.cache_data(show_spinner=False)
def load_sarimax_forecast(path: str = SARIMAX_FORECAST_PATH) -> pd.DataFrame | None:
    if not os.path.exists(path):
        st.warning("SARIMAX forecast file not found.")
        return None
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        if "type" not in df.columns:
            df["type"] = "forecast"
        return df
    except Exception as e:
        st.warning(f"Error loading SARIMAX forecast: {e}")
        return None

# =========================
# Load data / model / sarimax
# =========================
df = load_data()
model, preprocessor = load_model_and_preprocessor()
df_sarimax = load_sarimax_forecast()

kpi_names = df["english_name"].unique().tolist() if df is not None else []
correlation_matrix = None
if df is not None and not df.empty:
    try:
        df_pivot = df.pivot_table(index="date", columns="english_name", values="monthly_value")
        correlation_matrix = df_pivot.corr()
    except Exception as e:
        st.warning(f"Could not compute correlation matrix: {e}")

# =========================
# Forecasting (LGBM)
# =========================
def build_future_template(base_df: pd.DataFrame, future_date: pd.Timestamp) -> pd.DataFrame:
    grp_cols = ["account_id", "english_name", "dealer_code"]
    latest = base_df.sort_values("date").groupby(grp_cols, as_index=False).tail(1).copy()
    latest["year"] = future_date.year
    latest["month"] = future_date.month
    latest["date"] = future_date
    latest["monthly_value"] = np.nan
    if "yearly_value" in latest.columns:
        latest["yearly_value"] = np.nan
    latest["forecast"] = True
    return latest.reset_index(drop=True)

def predict_kpis(df_input: pd.DataFrame, model, preprocessor: dict, num_months: int = 3) -> pd.DataFrame:
    if model is None or preprocessor is None:
        st.error("Model/preprocessor not loaded.")
        return pd.DataFrame()
    work = df_input.copy()
    work["forecast"] = False
    future_blocks, last_known = [], work.copy()
    for _ in range(num_months):
        future_date = pd.to_datetime(last_known["date"].max()) + pd.DateOffset(months=1)
        future_df = build_future_template(last_known, future_date)
        combined = pd.concat([last_known, future_df], ignore_index=True)
        fe = create_features_and_encode(combined, preprocessor)
        mask_future = fe["date"] == future_date
        X_predict = fe.loc[mask_future, preprocessor["feature_cols"]].values
        y_hat_log = model.predict(X_predict)
        y_hat = np.expm1(y_hat_log)
        future_df["monthly_value"] = y_hat
        future_blocks.append(future_df)
        last_known = pd.concat([last_known, future_df], ignore_index=True)
    return pd.concat(future_blocks, ignore_index=True)

# =========================
# Sidebar Controls (Forecast)
# =========================
st.sidebar.header("User Controls")
forecast_steps = st.sidebar.number_input("Forecast horizon (months)", min_value=1, max_value=12, value=3, step=1)
if st.sidebar.button("Generate Forecast") and df is not None and model is not None and preprocessor is not None:
    with st.spinner("Generating forecast..."):
        df_forecast = predict_kpis(df, model, preprocessor, num_months=int(forecast_steps))
    st.session_state["forecast_data"] = df_forecast
    st.session_state["show_forecast"] = True
    st.session_state["show_what_if"] = False
elif "show_forecast" not in st.session_state:
    st.session_state["show_forecast"] = False
    st.session_state["show_what_if"] = False

# =========================
# KPI Visualization & Forecast (LGBM)
# =========================
st.header("KPI Visualization & Forecast (LGBM)")
selected_kpis = st.multiselect("Select KPIs to display", options=kpi_names,
                               default=[k for k in ["TOTAL SALES","TOTAL GROSS PROFIT"] if k in kpi_names])
if selected_kpis:
    base = df.copy() if df is not None else pd.DataFrame()
    base = base[base["english_name"].isin(selected_kpis)]
    if st.session_state.get("show_forecast", False):
        df_forecast = st.session_state.get("forecast_data", pd.DataFrame()).copy()
        df_forecast = df_forecast[df_forecast["english_name"].isin(selected_kpis)]
        base["forecast"] = False
        combined = pd.concat([base, df_forecast], ignore_index=True)
        fig = px.line(combined, x="date", y="monthly_value", color="english_name",
                      markers=True, title="KPI Values with Forecast (LGBM)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(base, x="date", y="monthly_value", color="english_name",
                      markers=True, title="KPI Values (LGBM)")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# KPI Visualization & Forecast (SARIMAX Dynamic)
# =========================
if df_sarimax is not None and not df_sarimax.empty:
    st.header("SARIMAX Forecast (Experimental)")
    
    kpi_options_sarimax = df_sarimax["english_name"].unique()
    selected_kpi_sarimax = st.selectbox("Select KPI (SARIMAX)", kpi_options_sarimax)

    kpi_data_sarimax = df_sarimax[df_sarimax["english_name"] == selected_kpi_sarimax].copy()
    if "type" not in kpi_data_sarimax.columns:
        kpi_data_sarimax["type"] = "forecast"
    if "monthly_value" not in kpi_data_sarimax.columns:
        kpi_data_sarimax["monthly_value"] = 0.0
    kpi_data_sarimax["date"] = pd.to_datetime(kpi_data_sarimax["date"])

    actual = kpi_data_sarimax[kpi_data_sarimax["type"]=="actual"]
    forecast = kpi_data_sarimax[kpi_data_sarimax["type"]=="forecast"]

    st.subheader("SARIMAX Forecast Table")
    st.dataframe(kpi_data_sarimax.sort_values("date"))

    fig = go.Figure()
    if not actual.empty:
        fig.add_trace(go.Scatter(x=actual["date"], y=actual["monthly_value"],
                                 mode="lines+markers", name="Actual", line=dict(color="blue")))
    if not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["monthly_value"],
                                 mode="lines+markers", name="Forecast", line=dict(color="orange", dash="dash")))
    fig.update_layout(title=f"SARIMAX Forecast for {selected_kpi_sarimax}",
                      xaxis_title="Date", yaxis_title="Monthly Value", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dynamic What-If Adjustment")
    adjustment = st.slider("Adjust forecast by %", -50, 50, 0, key=f"sarimax_adjust_{selected_kpi_sarimax}")
    
    if not forecast.empty:
        adjusted_forecast = forecast.copy()
        adjusted_forecast["monthly_value"] *= (1 + adjustment / 100.0)

        fig2 = go.Figure()
        if not actual.empty:
            fig2.add_trace(go.Scatter(x=actual["date"], y=actual["monthly_value"],
                                      mode="lines+markers", name="Actual", line=dict(color="blue")))
        fig2.add_trace(go.Scatter(x=forecast["date"], y=forecast["monthly_value"],
                                  mode="lines+markers", name="Original Forecast", line=dict(color="orange", dash="dash")))
        fig2.add_trace(go.Scatter(x=adjusted_forecast["date"], y=adjusted_forecast["monthly_value"],
                                  mode="lines+markers", name=f"Adjusted Forecast ({adjustment:+d}%)",
                                  line=dict(color="red", dash="dot")))
        fig2.update_layout(title=f"SARIMAX What-If Adjustment for {selected_kpi_sarimax}",
                           xaxis_title="Date", yaxis_title="Monthly Value", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# Correlation Heatmap
# =========================
st.header("KPI Correlation Heatmap")
if correlation_matrix is not None and not correlation_matrix.empty:
    try:
        top_correlated_kpis = correlation_matrix.abs().sum().nlargest(10).index
        filtered_corr_matrix = correlation_matrix.loc[top_correlated_kpis, top_correlated_kpis]
        fig_corr = px.imshow(filtered_corr_matrix, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r", title="Top 10 Correlated KPIs")
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render correlation heatmap: {e}")

# =========================
# Sidebar: Deliverables Checklist
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("### About this Dashboard")
st.sidebar.info(
    """
    **DealerScope AI – Predictive Analytics Dashboard**  
    - Forecasts KPI trends using **LightGBM** and **SARIMAX**  
    - Explores **correlations** across business metrics  
    - Provides an **interactive What-If analysis** to test scenarios  
    - Designed to support **data-driven decision making** in dealerships
    """
)
