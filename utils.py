"""
Utility functions & constants used across the project.
Edit column names here if your real dataset differs.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.cluster import KMeans

# ---- Canonical column names expected by the pipeline ----
COLS = {
    "order_id": "order_id",
    "timestamp": "order_ts",          # ISO datetime string
    "lat": "lat",
    "lon": "lon",
    "items": "items",
    "prep_time_min": "prep_time_min",
    "temperature_c": "temperature_c",
    "is_rain": "is_rain",             # 0/1
    "zone_id": "zone_id",             # optional in raw
}

def parse_ts(df: pd.DataFrame, ts_col: str = COLS["timestamp"]) -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df["date"] = df[ts_col].dt.date
    df["hour"] = df[ts_col].dt.hour
    df["dow"] = df[ts_col].dt.dayofweek  # 0=Mon
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def build_zones(df: pd.DataFrame, n_zones: int = 8, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If zone_id not present, create zones via K-Means on lat/lon.
    Returns (df_with_zones, zone_centers)
    """
    df = df.copy()
    if COLS["zone_id"] in df.columns and df[COLS["zone_id"]].notna().any():
        # Already has zones; compute centers
        centers = df.groupby(COLS["zone_id"])[[COLS["lat"], COLS["lon"]]].mean().reset_index().rename(
            columns={COLS["lat"]:"lat", COLS["lon"]:"lon"})
        return df, centers

    coords = df[[COLS["lat"], COLS["lon"]]].to_numpy()
    kmeans = KMeans(n_clusters=n_zones, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(coords)
    df[COLS["zone_id"]] = labels
    centers = pd.DataFrame({
        "zone_id": range(n_zones),
        "lat": kmeans.cluster_centers_[:,0],
        "lon": kmeans.cluster_centers_[:,1],
    })
    return df, centers

def zone_hour_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate orders to zone-hour level and create features for modeling demand.
    """
    df = parse_ts(df)
    group_cols = [COLS["zone_id"], "date", "hour"]
    agg = (
        df.groupby(group_cols)
          .agg(
              orders=("order_id", "count"),
              avg_items=("items", "mean"),
              avg_prep=("prep_time_min", "mean"),
              temperature_c=("temperature_c", "mean"),
              is_rain=("is_rain", "max"),
          )
          .reset_index()
    )
    # Time features
    agg["is_peak_hour"] = agg["hour"].isin([12,13,19,20,21]).astype(int)
    agg["sin_hour"] = np.sin(2*np.pi*agg["hour"]/24)
    agg["cos_hour"] = np.cos(2*np.pi*agg["hour"]/24)
    return agg

def split_train_test(df: pd.DataFrame, test_days: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple time-based split: last `test_days` worth of dates as test set.
    """
    all_dates = sorted(df["date"].unique())
    if len(all_dates) <= test_days:
        return df.iloc[:-1], df.iloc[-1:]
    split_date = all_dates[-test_days]
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    return train, test
