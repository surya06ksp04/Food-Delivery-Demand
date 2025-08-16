from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

FEATURES = ["hour","is_weekend","is_peak_hour","sin_hour","cos_hour",
            "avg_items","avg_prep","temperature_c","is_rain"]

def allocate_drivers(expected_orders: float, avg_service_time_min: float) -> int:
    return int(math.ceil(expected_orders * avg_service_time_min / 60.0))

def main(avg_service_time_min: float = 20.0):
    root = Path(__file__).resolve().parents[1]
    agg = pd.read_csv(root / "data" / "zone_hour_agg.csv", parse_dates=["date"])
    model = load(root / "models" / "demand_model.pkl")

    agg["pred_orders"] = model.predict(agg[FEATURES])
    # Driver allocation per zone-hour
    agg["drivers_needed"] = agg["pred_orders"].apply(lambda x: allocate_drivers(x, avg_service_time_min))

    # Peak hours plot (citywide)
    city_hr = agg.groupby(["date","hour"])["pred_orders"].sum().reset_index()
    out_img = root / "outputs" / "peak_hours.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)

    # Plotting total demand by hour averaged across days
    mean_by_hour = city_hr.groupby("hour")["pred_orders"].mean()
    plt.figure(figsize=(9,5))
    mean_by_hour.plot(kind="bar")
    plt.title("Average Predicted Orders by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Orders")
    plt.tight_layout()
    plt.savefig(out_img, dpi=160)
    plt.close()
    print("Saved", out_img)

    # Saving driver plan
    plan = agg[["zone_id","date","hour","pred_orders","drivers_needed"]]
    out_csv = root / "outputs" / "driver_plan.csv"
    plan.to_csv(out_csv, index=False)
    print("Saved", out_csv)

if __name__ == "__main__":
    main()
