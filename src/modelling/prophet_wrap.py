import pickle
import pandas as pd
from prophet import Prophet
from .base import ForecastModel

class ProphetWrapper(ForecastModel):
    def __init__(self, seasonality_mode: str = "additive"):
        # Initialize Prophet model with additive seasonality (default)
        self.model = Prophet(seasonality_mode=seasonality_mode)
        self.fitted = None

    def train(self, df: pd.DataFrame):
        """
        Fit Prophet model on input data.
        df must have columns ['ds', 'y'] where:
          - ds = datetime column
          - y = target series
        After training, store in-sample fitted values for evaluation.
        Returns the fitted model (not predictions).
        """
        self.model.fit(df)

        # Generate in-sample predictions to evaluate training fit
        forecast = self.model.predict(df)
        fitted_series = forecast[["ds", "yhat"]].set_index("ds")["yhat"]
        self.fitted = fitted_series

        return self.model

    def predict(self, steps: int) -> pd.Series:
        """
        Forecast 'steps' months ahead.
        Returns only the yhat predictions for those future steps.
        """
        future = self.model.make_future_dataframe(periods=steps, freq="MS")
        forecast = self.model.predict(future)
        preds = forecast[["ds", "yhat"]].set_index("ds")
        return preds.tail(steps)["yhat"]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
