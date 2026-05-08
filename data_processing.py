#Created by Ashton Foo

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class ProcessedData:
    raw_wide: pd.DataFrame
    long_data: pd.DataFrame
    feature_data: pd.DataFrame
    X_train_seq: np.ndarray
    X_test_seq: np.ndarray
    y_train_seq: np.ndarray
    y_test_seq: np.ndarray
    X_train_rf: np.ndarray
    X_test_rf: np.ndarray
    y_train_rf: np.ndarray
    y_test_rf: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: MinMaxScaler
    feature_columns: list


def load_scats_excel(scats_file):
    scats_file = Path(scats_file)

    if not scats_file.exists():
        raise FileNotFoundError(f"SCATS file not found: {scats_file}")

    df = pd.read_excel(scats_file, sheet_name="Data", header=1)
    df.columns = [str(c).strip() for c in df.columns]

    required_columns = {"SCATS Number", "Location", "Date"}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(f"Missing columns in SCATS data: {missing}")

    volume_columns = [c for c in df.columns if re.fullmatch(r"V\d{2}", c)]

    if len(volume_columns) != 96:
        raise ValueError(f"Expected 96 traffic volume columns V00-V95, found {len(volume_columns)}")

    df["SCATS Number"] = pd.to_numeric(df["SCATS Number"], errors="coerce").astype("Int64")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in volume_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["SCATS Number", "Date"])
    df[volume_columns] = df[volume_columns].interpolate(axis=1, limit_direction="both")

    return df


def wide_to_long(df):
    volume_columns = [c for c in df.columns if re.fullmatch(r"V\d{2}", c)]

    id_columns = [
        c for c in
        ["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE", "Date"]
        if c in df.columns
    ]

    long_df = df.melt(
        id_vars=id_columns,
        value_vars=volume_columns,
        var_name="slot",
        value_name="volume_15min"
    )

    long_df["slot_index"] = long_df["slot"].str[1:].astype(int)
    long_df["datetime"] = long_df["Date"] + pd.to_timedelta(long_df["slot_index"] * 15, unit="m")
    long_df["volume_15min"] = pd.to_numeric(long_df["volume_15min"], errors="coerce")

    long_df = long_df.dropna(subset=["volume_15min"])
    long_df = long_df.sort_values(["SCATS Number", "datetime"]).reset_index(drop=True)

    return long_df


def make_features(long_df):
    data = long_df.copy()

    data["hour"] = data["datetime"].dt.hour
    data["minute"] = data["datetime"].dt.minute
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["time_slot"] = data["hour"] * 4 + (data["minute"] // 15)

    data["slot_sin"] = np.sin(2 * np.pi * data["time_slot"] / 96)
    data["slot_cos"] = np.cos(2 * np.pi * data["time_slot"] / 96)

    grouped = data.groupby("SCATS Number", group_keys=False)

    for lag in [1, 2, 3, 4, 8, 12]:
        data[f"lag_{lag}"] = grouped["volume_15min"].shift(lag)

    data["rolling_mean_4"] = grouped["volume_15min"].shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    data["rolling_mean_8"] = grouped["volume_15min"].shift(1).rolling(8).mean().reset_index(level=0, drop=True)

    data["target_next_volume"] = grouped["volume_15min"].shift(-1)

    data = data.dropna().reset_index(drop=True)

    return data


def create_sequence_data(feature_df, feature_columns, sequence_length):
    X = []
    y = []

    for _, group in feature_df.groupby("SCATS Number"):
        group = group.sort_values("datetime")

        features = group[feature_columns].to_numpy(dtype=float)
        targets = group["target_next_volume"].to_numpy(dtype=float)

        for i in range(sequence_length, len(group)):
            X.append(features[i - sequence_length:i])
            y.append(targets[i])

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def prepare_data(scats_file, sequence_length=8, test_ratio=0.2):
    raw = load_scats_excel(scats_file)
    long_df = wide_to_long(raw)
    feature_df = make_features(long_df)

    feature_columns = [
        "SCATS Number",
        "hour",
        "day_of_week",
        "is_weekend",
        "time_slot",
        "slot_sin",
        "slot_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_8",
        "lag_12",
        "rolling_mean_4",
        "rolling_mean_8"
    ]

    split_time = feature_df["datetime"].quantile(1 - test_ratio)

    train_df = feature_df[feature_df["datetime"] <= split_time].copy()
    test_df = feature_df[feature_df["datetime"] > split_time].copy()

    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    train_df[feature_columns] = feature_scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = feature_scaler.transform(test_df[feature_columns])

    train_df["target_next_volume"] = target_scaler.fit_transform(train_df[["target_next_volume"]])
    test_df["target_next_volume"] = target_scaler.transform(test_df[["target_next_volume"]])

    X_train_seq, y_train_seq = create_sequence_data(train_df, feature_columns, sequence_length)
    X_test_seq, y_test_seq = create_sequence_data(test_df, feature_columns, sequence_length)

    X_train_rf = train_df[feature_columns].to_numpy(dtype=float)
    X_test_rf = test_df[feature_columns].to_numpy(dtype=float)
    y_train_rf = train_df["target_next_volume"].to_numpy(dtype=float)
    y_test_rf = test_df["target_next_volume"].to_numpy(dtype=float)

    return ProcessedData(
        raw_wide=raw,
        long_data=long_df,
        feature_data=feature_df,
        X_train_seq=X_train_seq,
        X_test_seq=X_test_seq,
        y_train_seq=y_train_seq,
        y_test_seq=y_test_seq,
        X_train_rf=X_train_rf,
        X_test_rf=X_test_rf,
        y_train_rf=y_train_rf,
        y_test_rf=y_test_rf,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=feature_columns
    )