import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_processing import load_scats_excel
from algorithms.travel_time import estimate_travel_time_seconds

CONNECTION_THRESHOLD_KM = 3.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_roads(location_str):
    for sep in (
        " N of ",
        " S of ",
        " E of ",
        " W of ",
        " NB ",
        " SB ",
        " EB ",
        " WB ",
        " N OF ",
        " S OF ",
        " E OF ",
        " W OF ",
    ):
        location_str = location_str.replace(sep, "/")

    return {
        p.strip().upper()
        for p in location_str.split("/")
        if p.strip()
    }


def share_road(loc_a, loc_b):
    return bool(_parse_roads(loc_a) & _parse_roads(loc_b))


def get_scats_sites(scats_file):
    df = load_scats_excel(scats_file)

    sites = (
        df[
            [
                "SCATS Number",
                "Location",
                "NB_LATITUDE",
                "NB_LONGITUDE",
            ]
        ]
        .drop_duplicates(subset=["SCATS Number"])
    )

    sites = sites.dropna(
        subset=["NB_LATITUDE", "NB_LONGITUDE"]
    )

    sites = sites[
        (sites["NB_LATITUDE"] != 0.0)
        & (sites["NB_LONGITUDE"] != 0.0)
    ]

    return {
        int(row["SCATS Number"]): {
            "lat": float(row["NB_LATITUDE"]),
            "lon": float(row["NB_LONGITUDE"]),
            "location": str(row["Location"]).strip(),
        }
        for _, row in sites.iterrows()
    }


def load_prediction_assets(models_dir="models"):
    models_dir = Path(models_dir)

    rf_model = joblib.load(
        models_dir / "random_forest_traffic_model.joblib"
    )

    feature_scaler = joblib.load(
        models_dir / "feature_scaler.joblib"
    )

    target_scaler = joblib.load(
        models_dir / "target_scaler.joblib"
    )

    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)

    return (
        rf_model,
        feature_scaler,
        target_scaler,
        feature_columns,
    )


def predict_flow(
    scats_number,
    time_slot,
    rf_model,
    feature_scaler,
    target_scaler,
    feature_columns,
):
    hour = (time_slot * 15) // 60

    feature_row = {
        "SCATS Number": scats_number,
        "hour": hour,
        "day_of_week": 0,
        "is_weekend": 0,
        "time_slot": time_slot,
        "slot_sin": np.sin(2 * np.pi * time_slot / 96),
        "slot_cos": np.cos(2 * np.pi * time_slot / 96),
        "lag_1": 50,
        "lag_2": 50,
        "lag_3": 50,
        "lag_4": 50,
        "lag_8": 50,
        "lag_12": 50,
        "rolling_mean_4": 50,
        "rolling_mean_8": 50,
    }

    X = pd.DataFrame([feature_row])

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    X_scaled = feature_scaler.transform(X)

    pred_scaled = rf_model.predict(X_scaled).reshape(-1, 1)

    pred = target_scaler.inverse_transform(
        pred_scaled
    ).ravel()[0]

    return max(0.0, float(pred))


def build_graph(scats_file, time_slot=32, models_dir="models"):
    (
        rf_model,
        feature_scaler,
        target_scaler,
        feature_columns,
    ) = load_prediction_assets(models_dir)

    nodes = get_scats_sites(scats_file)

    site_list = list(nodes.items())

    edges = {site: [] for site in nodes}

    K = 3

    for site_a, info_a in site_list:

        distances = []

        for site_b, info_b in site_list:

            if site_a == site_b:
                continue

            distance_km = haversine(
                info_a["lat"],
                info_a["lon"],
                info_b["lat"],
                info_b["lon"],
            )

            distances.append(
                (distance_km, site_b, info_b)
            )

        distances.sort(key=lambda x: x[0])

        nearest = distances[:K]

        for distance_km, site_b, info_b in nearest:

            flow_a = predict_flow(
                site_a,
                time_slot,
                rf_model,
                feature_scaler,
                target_scaler,
                feature_columns,
            )

            flow_b = predict_flow(
                site_b,
                time_slot,
                rf_model,
                feature_scaler,
                target_scaler,
                feature_columns,
            )

            predicted_flow = (flow_a + flow_b) / 2

            travel_time = estimate_travel_time_seconds(
                distance_km,
                predicted_flow,
            )

            if (
                (site_a == 970 and site_b == 2846)
                or
                (site_a == 2846 and site_b == 970)
            ):
                print("FORCE HUGE COST")
                travel_time = 999999

            if not any(
                neighbour == site_b
                for neighbour, _ in edges[site_a]
            ):
                edges[site_a].append(
                    (site_b, travel_time)
                )

            if not any(
                neighbour == site_a
                for neighbour, _ in edges[site_b]
            ):
                edges[site_b].append(
                    (site_a, travel_time)
                )

    total_edges = sum(len(v) for v in edges.values())

    isolated = [
        k for k, v in edges.items()
        if len(v) == 0
    ]
    print(f"Graph Built")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {total_edges}")
    print(f"Isolated nodes: {len(isolated)}")

    return nodes, edges