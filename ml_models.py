#Created by Ashton Foo

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_processing import prepare_data


def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM

        return tf, Sequential, EarlyStopping, Dense, Dropout, GRU, LSTM

    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required. Run: pip install -r requirements_ml.txt"
        ) from exc


def build_lstm(input_shape):
    _, Sequential, _, Dense, Dropout, _, LSTM = import_tensorflow()

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_gru(input_shape):
    _, Sequential, _, Dense, Dropout, GRU, _ = import_tensorflow()

    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def inverse_target(values, scaler):
    values = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(values).ravel()


def regression_metrics(y_true_scaled, y_pred_scaled, scaler):
    y_true = inverse_target(y_true_scaled, scaler)
    y_pred = inverse_target(y_pred_scaled, scaler)

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    return {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "MAPE_%": round(float(mape), 3),
        "R2": round(float(r2), 3)
    }


def train_all_models(scats_file, output_dir="models", sequence_length=8, epochs=25, batch_size=64):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Processing SCATS dataset...")
    processed = prepare_data(
        scats_file=scats_file,
        sequence_length=sequence_length
    )

    _, _, EarlyStopping, *_ = import_tensorflow()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    results = []

    print("\nTraining LSTM model...")
    lstm_model = build_lstm(
        input_shape=(processed.X_train_seq.shape[1], processed.X_train_seq.shape[2])
    )

    lstm_model.fit(
        processed.X_train_seq,
        processed.y_train_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    lstm_pred = lstm_model.predict(processed.X_test_seq, verbose=0).ravel()

    results.append({
        "Model": "LSTM",
        **regression_metrics(processed.y_test_seq, lstm_pred, processed.target_scaler)
    })

    lstm_model.save(output_dir / "lstm_traffic_model.keras")

    print("\nTraining GRU model...")
    gru_model = build_gru(
        input_shape=(processed.X_train_seq.shape[1], processed.X_train_seq.shape[2])
    )

    gru_model.fit(
        processed.X_train_seq,
        processed.y_train_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    gru_pred = gru_model.predict(processed.X_test_seq, verbose=0).ravel()

    results.append({
        "Model": "GRU",
        **regression_metrics(processed.y_test_seq, gru_pred, processed.target_scaler)
    })

    gru_model.save(output_dir / "gru_traffic_model.keras")

    print("\nTraining Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        processed.X_train_rf,
        processed.y_train_rf
    )

    rf_pred = rf_model.predict(processed.X_test_rf)

    results.append({
        "Model": "Random Forest",
        **regression_metrics(processed.y_test_rf, rf_pred, processed.target_scaler)
    })

    joblib.dump(rf_model, output_dir / "random_forest_traffic_model.joblib")
    joblib.dump(processed.feature_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(processed.target_scaler, output_dir / "target_scaler.joblib")

    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(processed.feature_columns, f, indent=2)

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(output_dir / "model_test_results.csv", index=False)

    print("\nModel testing results:")
    print(results_df.to_string(index=False))

    print(f"\nSaved models and results inside: {output_dir}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Train and test LSTM, GRU and Random Forest traffic prediction models."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the university SCATS October 2006 .xls file"
    )

    parser.add_argument(
        "--output",
        default="models",
        help="Folder to save trained models and test results"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8
    )

    args = parser.parse_args()

    train_all_models(
        scats_file=args.data,
        output_dir=args.output,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()