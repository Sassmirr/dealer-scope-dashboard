import numpy as np
import matplotlib.pyplot as plt

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def plot_forecast(dates, actual, predicted, title="Forecast vs Actual"):
    """Quick visualization of forecast"""
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual", marker="o")
    plt.plot(dates, predicted, label="Predicted", marker="x")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("KPI Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
