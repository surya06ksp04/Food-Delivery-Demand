"""
Simulate order + weather data for a city grid.
Generates `data/simulated_orders.csv`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

def simulate(n_days=14, city_center=(17.441, 78.389), city_radius_km=12, n_orders_per_day=3000):
    lat0, lon0 = city_center
    rows = []
    for d in range(n_days):
        # simple weather pattern
        temp_base = rng.normal(30, 2)
        is_rain_day = rng.random() < 0.25
        for i in range(n_orders_per_day):
            # time-of-day intensity (peaks at lunch 12-14 and dinner 19-22)
            hour = rng.integers(0, 24)
            intensity = (
                0.6 * np.exp(-((hour-13)**2)/(2*2.5**2)) +
                0.9 * np.exp(-((hour-20)**2)/(2*2.5**2)) +
                0.15
            )
            if rng.random() > intensity:  # skip many hours
                continue

            # random offset within hour
            minute = rng.integers(0, 60)
            ts = pd.Timestamp("2025-07-01") + pd.Timedelta(days=d, hours=hour, minutes=minute)

            # geo: sample around center within radius
            # rough conversion: 1 deg lat ~ 111km, lon scaled by cos(lat)
            r_km = rng.uniform(0, city_radius_km)
            theta = rng.uniform(0, 2*np.pi)
            dlat = (r_km/111.0) * np.cos(theta)
            dlon = (r_km/(111.0*np.cos(np.deg2rad(lat0)))) * np.sin(theta)
            lat = lat0 + dlat
            lon = lon0 + dlon

            items = max(1, int(rng.normal(2.2, 0.9)))
            prep = max(8, rng.normal(18, 5))
            temp = temp_base + rng.normal(0, 1.5)
            rain = int(is_rain_day and rng.random() < 0.6)

            rows.append({
                "order_id": f"O{d:02d}-{i:05d}",
                "order_ts": ts.isoformat(),
                "lat": lat,
                "lon": lon,
                "items": items,
                "prep_time_min": prep,
                "temperature_c": temp,
                "is_rain": rain
            })
    return pd.DataFrame(rows)

def main():
    df = simulate()
    out = Path(__file__).resolve().parents[1] / "data" / "simulated_orders.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")

if __name__ == "__main__":
    main()
