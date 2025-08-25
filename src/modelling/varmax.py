import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from .base import ForecastModel

class VARMAXModel(ForecastModel):
    def __init__(self, maxlags: int = 12):
        """Initialize VAR model wrapper."""
        self.maxlags = maxlags
        self.model = None
        self.results = None

    def train(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        """
        Fit a VAR model on wide-format data (date as index, KPIs as columns).
        Returns in-sample predictions aligned with the input for evaluation.
        """
        df_wide = df_wide.astype(float)
        n_obs = len(df_wide)

        # Ensure maxlags does not exceed available data
        usable_lags = max(1, min(self.maxlags, n_obs // 4))

        # Fit VAR model
        self.model = VAR(df_wide)
        self.results = self.model.fit(maxlags=usable_lags, ic="aic")

        # Get in-sample fitted values for evaluation
        fitted_vals = self.results.fittedvalues
        preds = pd.DataFrame(fitted_vals, index=df_wide.index, columns=df_wide.columns)
        return preds

    def predict(self, steps: int = 3) -> np.ndarray:
        """
        Forecast future steps using the fitted VAR model.
        """
        if self.results is None:
            raise ValueError("Model not trained yet.")
        k_ar = self.results.k_ar
        last_obs = self.results.endog[-k_ar:]
        return self.results.forecast(last_obs, steps=steps)

    def save(self, path: str):
        """
        Save trained VAR model results to disk.
        """
        with open(path, "wb") as f:
            pickle.dump(self.results, f)

    def load(self, path: str):
        """
        Load previously trained VAR model results from disk.
        """
        with open(path, "rb") as f:
            self.results = pickle.load(f)
