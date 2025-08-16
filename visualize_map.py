from __future__ import annotations
from pathlib import Path
import pandas as pd
import folium

def main(target_hour: int = 20):
    root = Path(__file__).resolve().parents[1]
    centers = pd.read_csv(root / "data" / "zone_centers.csv")
    plan = pd.read_csv(root / "outputs" / "driver_plan.csv")
    latest_date = plan["date"].max()
    subset = plan[(plan["date"]==latest_date) & (plan["hour"]==target_hour)]
    merged = centers.merge(subset, on="zone_id", how="left")

    mean_lat = merged["lat"].mean()
    mean_lon = merged["lon"].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

    max_orders = merged["pred_orders"].max()
    for _, r in merged.iterrows():
        demand = float(r.get("pred_orders", 0) or 0.0)
        intensity = 0.0 if max_orders == 0 else demand / max_orders
        color = "#{:02x}{:02x}{:02x}".format(int(intensity*255), 0, 0)
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=12,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=(f"Zone {int(r['zone_id'])}<br>"
                   f"Predicted Orders: {demand:.1f}<br>"
                   f"Drivers Needed: {int(r.get('drivers_needed',0))}")
        ).add_to(m)

    out_html = root / "outputs" / "demand_map.html"
    m.save(out_html)
    print("Saved", out_html)

if __name__ == "__main__":
    main()
