import pickle
import pandas as pd
import lightgbm as lgb
from .base import ForecastModel

class LGBMModel(ForecastModel):
    def __init__(self):
        self.model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        """
        Fit LightGBM and return in-sample predictions aligned with training index.
        """
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
        self.model = lgb.train(params, train_data, num_boost_round=100)
        
        preds = pd.Series(self.model.predict(X_train), index=X_train.index)
        return preds

    def predict(self, X_future: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return pd.Series(self.model.predict(X_future), index=X_future.index)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
