from __future__ import annotations

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump
from utils import split_train_test

FEATURES = ["hour","is_weekend","is_peak_hour","sin_hour","cos_hour",
            "avg_items","avg_prep","temperature_c","is_rain"]
TARGET = "orders"

def main():
    root = Path(__file__).resolve().parents[1]
    agg = pd.read_csv(root / "data" / "zone_hour_agg.csv", parse_dates=["date"])
    train, test = split_train_test(agg, test_days=3)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R^2:", r2_score(y_test, preds))

    out = root / "models" / "demand_model.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    print("Saved model to", out)

if __name__ == "__main__":
    main()
