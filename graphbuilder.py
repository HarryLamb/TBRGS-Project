# Created by Harry Lamb

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_processing import load_scats_excel, wide_to_long
from algorithms.travel_time import estimate_travel_time_seconds

K_NEAREST = 5
FIXED_HOUR = 15
FIXED_SLOT = 60
FIXED_DAY = 4
FIXED_IS_WEEKEND = 0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_scats_sites(scats_file):
    df = load_scats_excel(scats_file)
    sites = (df[["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE"]]
             .drop_duplicates(subset=["SCATS Number"]))
    sites = sites.dropna(subset=["NB_LATITUDE", "NB_LONGITUDE"])
    sites = sites[(sites["NB_LATITUDE"] != 0.0) & (sites["NB_LONGITUDE"] != 0.0)]
    return {
        int(row["SCATS Number"]): {
            "lat":      float(row["NB_LATITUDE"]),
            "lon":      float(row["NB_LONGITUDE"]),
            "location": str(row["Location"]).strip(),
        }
        for _, row in sites.iterrows()
    }


def build_flow_lookup(scats_file):
    df = load_scats_excel(scats_file)
    long_df = wide_to_long(df)
    long_df["slot"] = long_df["datetime"].dt.hour * 4 + long_df["datetime"].dt.minute // 15
    lookup = (
        long_df.groupby(["SCATS Number", "slot"])["volume_15min"]
        .mean()
        .to_dict()
    )
    return lookup


def load_prediction_assets(models_dir="models"):
    models_dir = Path(models_dir)
    rf_model       = joblib.load(models_dir / "random_forest_traffic_model.joblib")
    feature_scaler = joblib.load(models_dir / "feature_scaler.joblib")
    target_scaler  = joblib.load(models_dir / "target_scaler.joblib")
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return rf_model, feature_scaler, target_scaler, feature_columns


def predict_flow(scats_number, flow_lookup, rf_model, feature_scaler, target_scaler, feature_columns):
    def get_lag(lag):
        slot = (FIXED_SLOT - lag) % 96
        return flow_lookup.get((scats_number, slot), 50.0)

    feature_row = {
        "SCATS Number":   0,
        "hour":           FIXED_HOUR,
        "day_of_week":    FIXED_DAY,
        "is_weekend":     FIXED_IS_WEEKEND,
        "time_slot":      FIXED_SLOT,
        "slot_sin":       np.sin(2 * np.pi * FIXED_SLOT / 96),
        "slot_cos":       np.cos(2 * np.pi * FIXED_SLOT / 96),
        "lag_1":          get_lag(1),
        "lag_2":          get_lag(2),
        "lag_3":          get_lag(3),
        "lag_4":          get_lag(4),
        "lag_8":          get_lag(8),
        "lag_12":         get_lag(12),
        "rolling_mean_4": sum(get_lag(i) for i in range(1, 5)) / 4,
        "rolling_mean_8": sum(get_lag(i) for i in range(1, 9)) / 8,
    }

    X = pd.DataFrame([feature_row])[feature_columns]
    X_scaled = feature_scaler.transform(X)
    pred_scaled = rf_model.predict(X_scaled).reshape(-1, 1)
    pred = target_scaler.inverse_transform(pred_scaled).ravel()[0]
    return max(0.0, float(pred))


def build_graph(scats_file, models_dir="models"):
    rf_model, feature_scaler, target_scaler, feature_columns = load_prediction_assets(models_dir)
    nodes = get_scats_sites(scats_file)
    flow_lookup = build_flow_lookup(scats_file)
    site_list = list(nodes.items())
    edges = {site: [] for site in nodes}

    for site_a, info_a in site_list:
        distances = sorted(
            [
                (haversine(info_a["lat"], info_a["lon"], info_b["lat"], info_b["lon"]), site_b, info_b)
                for site_b, info_b in site_list if site_b != site_a
            ],
            key=lambda x: x[0]
        )
        for distance_km, site_b, _ in distances[:K_NEAREST]:
            predicted_flow = predict_flow(
                site_a, flow_lookup,
                rf_model, feature_scaler, target_scaler, feature_columns
            )
            travel_time = estimate_travel_time_seconds(distance_km, predicted_flow)
            if not any(n == site_b for n, _ in edges[site_a]):
                edges[site_a].append((site_b, travel_time))
            if not any(n == site_a for n, _ in edges[site_b]):
                edges[site_b].append((site_a, travel_time))

    return nodes, edges