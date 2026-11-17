"""Inference logic for the Crop Recommendation model."""
from __future__ import annotations

import numpy as np

from utils.exceptions import InferenceError
from utils.logger import get_logger
from utils.model_loader import ModelRegistry

logger = get_logger(__name__)

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "moisture", "ph"]


def predict_crop(
    n: float, p: float, k: float, temperature: float,
    humidity: float, moisture: float, ph: float,
) -> dict:
    """
    Predict the most suitable crop given soil/environmental readings.

    `humidity` and `ph` are expected to already be resolved by the caller
    (either a real sensor reading, or an auto-filled value -- see
    services/weather_fetch_service.py for how the API layer fills these in
    when the IoT device lacks a humidity or pH sensor).

    Returns:
        {
          "recommended_crop": str,
          "confidence": float,
          "top_3": [{"crop": str, "probability": float}, ...]
        }
    """
    try:
        model = ModelRegistry.get("crop_recommendation_model")
        scaler = ModelRegistry.get("crop_recommendation_scaler")
        encoder = ModelRegistry.get("crop_recommendation_encoder")

        features = np.array([[n, p, k, temperature, humidity, moisture, ph]])
        features_scaled = scaler.transform(features)

        probabilities = model.predict_proba(features_scaled)[0]
        pred_idx = int(np.argmax(probabilities))
        pred_label = encoder.inverse_transform([pred_idx])[0]

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_3 = [
            {"crop": encoder.inverse_transform([i])[0], "probability": round(float(probabilities[i]), 4)}
            for i in top_indices
        ]

        return {
            "recommended_crop": pred_label,
            "confidence": round(float(probabilities[pred_idx]), 4),
            "top_3": top_3,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Crop prediction failed: {exc}")
        raise InferenceError(f"Crop prediction failed: {exc}") from exc
