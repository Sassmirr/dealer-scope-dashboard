import joblib

files = [
    "models/rf_account_0960P.joblib",  # Best
    "models/rf_account_0962P.joblib",  # Average
    "models/rf_account_0963P.joblib",  # Worst
]

for f in files:
    model_info = joblib.load(f)
    rf = model_info["model"]
    feat_cols = model_info["feature_cols"]
    print(f"\n=== {f} ===")
    for col, imp in sorted(zip(feat_cols, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"{col:<20} {imp:.4f}")
