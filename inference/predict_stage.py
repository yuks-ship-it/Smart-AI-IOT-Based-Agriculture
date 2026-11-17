"""Inference logic for the Crop Growth Stage classification model."""
from __future__ import annotations

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from preprocessing.image_preprocessing import decode_and_resize_raw, to_batch
from utils.config_loader import ConfigLoader
from utils.exceptions import InferenceError
from utils.logger import get_logger
from utils.model_loader import ModelRegistry


def predict_stage(image_bytes: bytes, crop_type: str) -> dict:
    """
    Predict the growth stage of a crop from an image.

    The preprocessing matches training: raw 0-255 image is decoded,
    resized, then passed through MobileNetV2's preprocess_input (which
    normalizes to [-1, 1] range).

    Args:
        image_bytes: raw bytes of the uploaded image
        crop_type: one of the supported crops (rice/wheat/maize)

    Returns:
        {"crop": str, "growth_stage": str, "confidence": float,
         "all_probabilities": {...}, "low_confidence_warning": bool}
    """
    logger = get_logger(__name__)
    try:
        settings = ConfigLoader.load("app_settings.json")
        img_size = tuple(settings["models"]["stage_classification"]["image_size"])

        model = ModelRegistry.get("stage_model")
        labels = ModelRegistry.get_metadata("stage_labels")
        if not labels:
            raise InferenceError("Stage labels metadata not loaded.")

        # decode_and_resize_raw returns float32 in [0, 255] range — same as training
        img = decode_and_resize_raw(image_bytes, target_size=img_size)
        img = preprocess_input(img)  # MobileNetV2 normalization: scales to [-1, 1]
        batch = to_batch(img)

        predictions = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(predictions))
        pred_label = labels[str(pred_idx)] if str(pred_idx) in labels else labels[pred_idx]

        confidence_threshold = settings.get("confidence_threshold", 0.5)
        confidence = float(predictions[pred_idx])

        all_probs = {
            (labels[str(i)] if str(i) in labels else labels[i]): round(float(p), 4)
            for i, p in enumerate(predictions)
        }

        return {
            "crop": crop_type,
            "growth_stage": pred_label,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "low_confidence_warning": confidence < confidence_threshold,
        }
    except InferenceError:
        raise
    except Exception as exc:
        logger.error(f"Stage prediction failed: {exc}")
        raise InferenceError(f"Stage prediction failed: {exc}") from exc
