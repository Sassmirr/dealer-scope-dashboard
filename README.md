# 🚗 DealerScope AI – Predictive Analytics Dashboard

DealerScope AI is an **interactive Streamlit dashboard** for automotive KPI forecasting.  
It combines **machine learning (LightGBM)** with **classical time series modeling (SARIMAX)** to provide accurate, explainable, and flexible predictions.  
A **What-If scenario engine** lets business users simulate changes (e.g. Sales ±50%) and instantly see **how other KPIs like Gross Profit will respond.**
---

## ✨ Features
- **Accurate KPI Forecasting**  
  Predicts the next 1–12 months of performance using a trained LightGBM model.
  
- **Correlation-Driven What-If Analysis**  
  Adjust key metrics (e.g. Sales ±50%) and see how **Gross Profit responds dynamically**, powered by real historical regression — not random scaling.
  
- **Interactive KPI Dashboard**  
  Live charts, KPI cards, and dealer-level insights all update instantly in response to user input.
  
- **Clean & Modular Codebase**  
  Organized into `src/models/` for machine learning, `data/` for inputs, and a single `app.py` for Streamlit deployment.

---

## 📊 Tech Stack
- **[Python 3.10+](https://www.python.org/)**  
- **[LightGBM](https://lightgbm.readthedocs.io/)** for fast gradient boosting  
- **[Streamlit](https://streamlit.io/)** for dashboard UI  
- **[Plotly](https://plotly.com/python/)** for interactive charts  
- **[scikit-learn](https://scikit-learn.org/)** for regression modeling in What-If engine  

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Sassmirr/dealer-scope-dashboard.git
   cd dealer-scope-dashboard
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # for Linux/Mac
   venv\Scripts\activate      # for Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
Run the application:
```bash
python app.py
```
Then open your browser and navigate to:
```
http://localhost:5000
```
*(If using Streamlit: `streamlit run app.py`)*


🧠 How the Forecasting & What-If Engine Works

Forecast Generation

Choose LightGBM or SARIMAX to predict future KPIs.

Forecast horizon: 1–12 months.

What-If Analysis

Move a slider to increase/decrease TOTAL SALES by %.

TOTAL GROSS PROFIT is recalculated automatically using a historical linear regression between Sales and Profit.

All charts update in real time to reflect realistic financial outcomes.

