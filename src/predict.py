import pandas as pd
import argparse
import os
import joblib
from prophet import Prophet
from statsmodels.tsa.statespace.varmax import VARMAX
import lightgbm as lgb

def prepare_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_prophet_model(df, model_path):
    if os.path.exists(model_path):
        print("Loading saved Prophet model...")
        model = joblib.load(model_path)
    else:
        print("Training new Prophet model...")
        df_prophet = df[['date','target']].rename(columns={'date':'ds','target':'y'})
        model = Prophet()
        model.fit(df_prophet)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
    return model

def predict_prophet(df, model_path, steps=10):
    model = get_prophet_model(df, model_path)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast[['ds','yhat']].tail(steps).reset_index(drop=True)

def get_varmax_model(df, model_path):
    if os.path.exists(model_path):
        print("Loading saved VARMAX model...")
        model_fit = joblib.load(model_path)
    else:
        print("Training new VARMAX model...")
        df_varmax = df.set_index('date')[['target']]
        model = VARMAX(df_varmax, order=(1,1))
        model_fit = model.fit(disp=False)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_fit, model_path)
    return model_fit

def predict_varmax(df, model_path, steps=10):
    model_fit = get_varmax_model(df, model_path)
    forecast = model_fit.forecast(steps=steps)
    return forecast.reset_index(drop=True)

def get_lgb_model(df, model_path):
    if os.path.exists(model_path):
        print("Loading saved LightGBM model...")
        model = joblib.load(model_path)
    else:
        print("Training new LightGBM model...")
        df['lag_1'] = df['target'].shift(1)
        df['lag_2'] = df['target'].shift(2)
        df.dropna(inplace=True)
        X = df[['lag_1','lag_2']]
        y = df['target']
        model = lgb.LGBMRegressor()
        model.fit(X, y)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
    return model

def predict_lgb(df, model_path, steps=10):
    model = get_lgb_model(df, model_path)
    # Prepare last rows for forecasting
    df_lag = df[['target']].copy()
    df_lag['lag_1'] = df_lag['target'].shift(1)
    df_lag['lag_2'] = df_lag['target'].shift(2)
    df_lag.dropna(inplace=True)
    last_rows = df_lag.tail(2).copy()
    preds = []
    for _ in range(steps):
        x_pred = last_rows[['lag_1','lag_2']].values[-1].reshape(1,-1)
        y_pred = model.predict(x_pred)[0]
        preds.append(y_pred)
        last_rows = pd.concat([last_rows, pd.DataFrame({'target':[y_pred],
                                                        'lag_1':[last_rows['lag_2'].values[-1]],
                                                        'lag_2':[y_pred]})], ignore_index=True)
    return pd.DataFrame(preds, columns=['lgb_pred'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    df = prepare_data(args.input)

    # Define model paths
    prophet_path = os.path.join(args.models, "prophet_model.pkl")
    varmax_path = os.path.join(args.models, "varmax_model.pkl")
    lgb_path = os.path.join(args.models, "lgb_model.pkl")

    # Generate predictions
    forecast_prophet = predict_prophet(df, prophet_path, args.steps)
    forecast_varmax = predict_varmax(df, varmax_path, args.steps)
    forecast_lgb = predict_lgb(df, lgb_path, args.steps)

    # Combine forecasts
    output_df = pd.concat([forecast_prophet.reset_index(drop=True),
                           forecast_varmax.reset_index(drop=True),
                           forecast_lgb.reset_index(drop=True)], axis=1)

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Forecasts saved to {args.output}")
