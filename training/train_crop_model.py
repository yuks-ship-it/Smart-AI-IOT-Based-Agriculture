"""
Train the Crop Recommendation model.

Input features: N, P, K, temperature, humidity, moisture
Target: crop label (rice / wheat / maize / ... as present in the dataset)

Usage:
    python training/train_crop_model.py
    python training/train_crop_model.py --algorithm xgboost
    python training/train_crop_model.py --data datasets/sensor_data/my_real_data.csv

Run this from the AI/ project root (or it will still resolve paths correctly
since paths are anchored to the project root via utils.config_loader).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config_loader import ConfigLoader, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "moisture", "ph"]
TARGET_COLUMN = "label"
DEFAULT_DATA_PATH = "datasets/sensor_data/crop_recommendation.csv"


def load_dataset(data_path: str) -> pd.DataFrame:
    path = resolve_path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Place your CSV at {data_path} with columns: "
            f"{FEATURE_COLUMNS + [TARGET_COLUMN]}"
        )
    df = pd.read_csv(path)

    # ph is treated as optional in the source CSV: if your dataset/device has
    # no pH sensor, we fill the configured neutral default so older datasets
    # (collected before pH support was added) still train without edits.
    if "ph" not in df.columns:
        from utils.config_loader import ConfigLoader
        default_ph = ConfigLoader.load("app_settings.json")["sensor_fallbacks"]["ph"]["fallback_value"]
        logger.warning(
            f"Dataset has no 'ph' column. Filling with default value {default_ph} for all rows. "
            f"Add a real 'ph' column to your CSV for better accuracy if you have pH sensor data."
        )
        df["ph"] = default_ph

    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    logger.info(f"Loaded dataset with shape {df.shape} from {path}")
    return df


def build_model(algorithm: str):
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )
    elif algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Run `pip install xgboost` or use "
                "--algorithm random_forest instead."
            ) from exc
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser(description="Train crop recommendation model.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--algorithm", type=str, default="random_forest",
        choices=["random_forest", "xgboost"],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_dataset(args.data)

    X = df[FEATURE_COLUMNS].values
    y_raw = df[TARGET_COLUMN].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    logger.info(f"Classes: {list(encoder.classes_)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=42, stratify=y
    )

    model = build_model(args.algorithm)
    logger.info(f"Training {args.algorithm} model on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    settings = ConfigLoader.load("app_settings.json")
    model_cfg = settings["models"]["crop_recommendation"]

    model_path = resolve_path(model_cfg["path"])
    scaler_path = resolve_path(model_cfg["scaler_path"])
    encoder_path = resolve_path(model_cfg["label_encoder_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)

    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved scaler to {scaler_path}")
    logger.info(f"Saved label encoder to {encoder_path}")

    from utils.model_versioning import register_version

    version_id = register_version(
        model_name="crop_recommendation",
        live_file_paths=[model_cfg["path"], model_cfg["scaler_path"], model_cfg["label_encoder_path"]],
        metrics={"test_accuracy": round(acc, 4)},
        algorithm=args.algorithm,
    )
    logger.info(f"Registered as version {version_id}. Run `python training/list_model_versions.py` to view history.")
    logger.info("Crop recommendation training complete.")


if __name__ == "__main__":
    main()
