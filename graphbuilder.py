# Created by Harry Lamb
# Machine Learning Parts created by Ashton Foo

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_processing import load_scats_excel, wide_to_long
from algorithms.travel_time import estimate_travel_time_seconds

# Number of nearest neighbours to connect each SCATS site to
K_NEAREST = 5
# Fixed to Friday for all predictions
FIXED_DAY = 4


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_scats_sites(scats_file):
    # Extract unique SCATS site locations from the dataset
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
    # Build a table of average traffic volume per site per 15 minutes for lag
    df = load_scats_excel(scats_file)
    long_df = wide_to_long(df)
    long_df["slot"] = long_df["datetime"].dt.hour * 4 + long_df["datetime"].dt.minute // 15
    lookup = (
        long_df.groupby(["SCATS Number", "slot"])["volume_15min"]
        .mean()
        .to_dict()
    )
    return lookup


def load_prediction_assets(models_dir="models", model_type="random_forest"):
    # Load the trained ML model, scalers and feature column list
    models_dir = Path(models_dir)
    feature_scaler = joblib.load(models_dir / "feature_scaler.joblib")
    target_scaler  = joblib.load(models_dir / "target_scaler.joblib")
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    if model_type == "lstm":
        import tensorflow as tf
        model = tf.keras.models.load_model(models_dir / "lstm_traffic_model.keras")
        model_type_loaded = "deep"
    elif model_type == "gru":
        import tensorflow as tf
        model = tf.keras.models.load_model(models_dir / "gru_traffic_model.keras")
        model_type_loaded = "deep"
    else:
        model = joblib.load(models_dir / "random_forest_traffic_model.joblib")
        model_type_loaded = "rf"
    return model, model_type_loaded, feature_scaler, target_scaler, feature_columns


def build_sequence(scats_number, flow_lookup, feature_columns,
                   feature_scaler, sequence_length=8, time_slot=60):
    # Build a sequence of feature rows for LSTM/GRU input
    sequence = []
    for step in range(sequence_length, 0, -1):
        slot = (time_slot - step) % 96

        def get_lag_for_slot(lag, s=slot):
            return flow_lookup.get((scats_number, (s - lag) % 96), 50.0)

        row = {
            "SCATS Number":   0,
            "hour":           (slot * 15) // 60,
            "day_of_week":    FIXED_DAY,
            "is_weekend":     0,
            "time_slot":      slot,
            "slot_sin":       np.sin(2 * np.pi * slot / 96),
            "slot_cos":       np.cos(2 * np.pi * slot / 96),
            "lag_1":          get_lag_for_slot(1),
            "lag_2":          get_lag_for_slot(2),
            "lag_3":          get_lag_for_slot(3),
            "lag_4":          get_lag_for_slot(4),
            "lag_8":          get_lag_for_slot(8),
            "lag_12":         get_lag_for_slot(12),
            "rolling_mean_4": sum(get_lag_for_slot(i) for i in range(1, 5)) / 4,
            "rolling_mean_8": sum(get_lag_for_slot(i) for i in range(1, 9)) / 8,
        }
        sequence.append(pd.DataFrame([row])[feature_columns].values[0])

    X_scaled = feature_scaler.transform(np.array(sequence))
    return X_scaled[np.newaxis, :, :]


def predict_flow(scats_number, flow_lookup, model, model_type_loaded,
                 feature_scaler, target_scaler, feature_columns,
                 sequence_length=8, time_slot=60):
    # Predict 15 minute traffic flow at a site for the certain time
    hour = (time_slot * 15) // 60

    def get_lag(lag):
        slot = (time_slot - lag) % 96
        return flow_lookup.get((scats_number, slot), 50.0)

    if model_type_loaded == "deep":
        X_seq = build_sequence(
            scats_number, flow_lookup, feature_columns,
            feature_scaler, sequence_length, time_slot=time_slot
        )
        pred_scaled = model.predict(X_seq, verbose=0).ravel()
    else:
        feature_row = {
            "SCATS Number":   0,
            "hour":           hour,
            "day_of_week":    FIXED_DAY,
            "is_weekend":     0,
            "time_slot":      time_slot,
            "slot_sin":       np.sin(2 * np.pi * time_slot / 96),
            "slot_cos":       np.cos(2 * np.pi * time_slot / 96),
            "lag_1":          get_lag(1),
            "lag_2":          get_lag(2),
            "lag_3":          get_lag(3),
            "lag_4":          get_lag(4),
            "lag_8":          get_lag(8),
            "lag_12":         get_lag(12),
            "rolling_mean_4": sum(get_lag(i) for i in range(1, 5)) / 4,
            "rolling_mean_8": sum(get_lag(i) for i in range(1, 9)) / 8,
        }
        X_row = pd.DataFrame([feature_row])[feature_columns]
        X_scaled = feature_scaler.transform(X_row)
        pred_scaled = model.predict(X_scaled).reshape(-1, 1)

    pred = target_scaler.inverse_transform(
        np.array(pred_scaled).reshape(-1, 1)
    ).ravel()[0]
    return max(0.0, float(pred))


def build_graph(scats_file, models_dir="models", model_type="random_forest", time_slot=60):
    # Build the road network graph for Boroondara, with edge costs being the ML prediction
    model, model_type_loaded, feature_scaler, target_scaler, feature_columns = (
        load_prediction_assets(models_dir, model_type)
    )
    nodes = get_scats_sites(scats_file)
    flow_lookup = build_flow_lookup(scats_file)
    site_list = list(nodes.items())
    edges = {site: [] for site in nodes}

    for site_a, info_a in site_list:
        # Sort all other sites by distance from site_a
        distances = sorted(
            [
                (haversine(info_a["lat"], info_a["lon"], info_b["lat"], info_b["lon"]), site_b, info_b)
                for site_b, info_b in site_list if site_b != site_a
            ],
            key=lambda x: x[0]
        )
        # Connect to K nearest neighbours
        for distance_km, site_b, _ in distances[:K_NEAREST]:
            predicted_flow = predict_flow(
                site_a, flow_lookup, model, model_type_loaded,
                feature_scaler, target_scaler, feature_columns,
                time_slot=time_slot
            )
            travel_time = estimate_travel_time_seconds(distance_km, predicted_flow)
            if not any(n == site_b for n, _ in edges[site_a]):
                edges[site_a].append((site_b, travel_time))
            if not any(n == site_a for n, _ in edges[site_b]):
                edges[site_b].append((site_a, travel_time))

    return nodes, edges