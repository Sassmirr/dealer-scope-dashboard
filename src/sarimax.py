import pandas as pd
import warnings
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def manual_sarimax(series, exog, steps=3):
    """Manual SARIMAX fallback if auto_arima fails, with exogenous regressors."""
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    m = 12  # monthly seasonality

    best_aic = float("inf")
    best_model = None

    for i in p:
        for j in d:
            for k in q:
                for I in P:
                    for J in D:
                        for K in Q:
                            try:
                                model = SARIMAX(
                                    series,
                                    exog=exog,
                                    order=(i, j, k),
                                    seasonal_order=(I, J, K, m),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                res = model.fit(disp=False)
                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_model = res
                            except Exception:
                                continue

    if best_model:
        # Forecast with future exogenous variables (repeat last row if not available)
        if exog is not None and len(exog) > 0:
            future_exog = pd.DataFrame([exog.iloc[-1].values] * steps, columns=exog.columns)
        else:
            future_exog = None
        forecast = best_model.forecast(steps=steps, exog=future_exog)
        return forecast
    return None


def predict_sarimax_auto(df, steps=3):
    results = []

    for kpi, group in df.groupby("english_name"):
        group = group.sort_values("date").copy()
        series = group["monthly_value"]
        # Exogenous regressors → account_id, year, month
        exog = group[["account_id", "year", "month"]]

        print(f"\nProcessing KPI: {kpi} ({len(series)} data points)")

        # Add actual data first
        for d, val in zip(group["date"], series):
            results.append({
                "english_name": kpi,
                "date": d,
                "monthly_value": float(val),
                "type": "actual"
            })

        pred = None
        # Try auto_arima first with exogenous regressors
        try:
            model = auto_arima(
                series,
                exogenous=exog,
                seasonal=True,
                m=12,
                trace=False,
                suppress_warnings=True,
                stepwise=True
            )
            # Forecast → use last exog row repeated
            future_exog = pd.DataFrame([exog.iloc[-1].values] * steps, columns=exog.columns)
            pred = model.predict(n_periods=steps, exogenous=future_exog)
            print(f"Auto ARIMA succeeded for {kpi} → params: {model.order}, seasonal: {model.seasonal_order}")
        except Exception as e:
            print(f"Auto ARIMA failed for {kpi}: {e}")

        # If auto_arima failed → try manual SARIMAX
        if pred is None:
            try:
                pred = manual_sarimax(series, exog, steps)
                if pred is not None:
                    print(f"Manual SARIMAX succeeded for {kpi}")
            except Exception as e:
                print(f"Manual SARIMAX failed for {kpi}: {e}")

        # If both failed → fallback to flat forecast
        if pred is None:
            pred = [series.iloc[-1]] * steps
            print(f"Fallback flat forecast used for {kpi}")

        # Create future dates starting from next month
        dates = pd.date_range(
            start=group["date"].max() + pd.offsets.MonthBegin(),
            periods=steps,
            freq="MS"
        )

        for d, val in zip(dates, pred):
            results.append({
                "english_name": kpi,
                "date": d,
                "monthly_value": float(val),
                "type": "forecast"
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])

    print(f"Loaded dataset with {len(df)} rows and {df['english_name'].nunique()} KPIs")

    # Run SARIMAX forecasts for 3 months ahead
    forecast_df = predict_sarimax_auto(df, steps=3)

    print("\n=== Auto SARIMAX Forecast Results (with history) ===")
    print(forecast_df.head(20))

    # Save to CSV
    forecast_df.to_csv("src/outputs/sarimax_forecast2.csv", index=False)
    print("Forecast saved to src/outputs/sarimax_forecast2.csv")
