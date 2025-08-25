from abc import ABC, abstractmethod

class ForecastModel(ABC):
    """Abstract interface all forecasting models must follow."""

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
