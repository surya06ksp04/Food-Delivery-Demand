from __future__ import annotations
import pandas as pd
from pathlib import Path
from utils import COLS, build_zones, zone_hour_aggregate

def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    raw_csv = data_dir / "simulated_orders.csv"
    df = pd.read_csv(raw_csv)
    df_z, centers = build_zones(df, n_zones=8, random_state=42)
    centers.to_csv(data_dir / "zone_centers.csv", index=False)

    agg = zone_hour_aggregate(df_z)
    agg.to_csv(data_dir / "zone_hour_agg.csv", index=False)
    print("Saved:")
    print(" -", data_dir / "zone_centers.csv")
    print(" -", data_dir / "zone_hour_agg.csv")

if __name__ == "__main__":
    main()
