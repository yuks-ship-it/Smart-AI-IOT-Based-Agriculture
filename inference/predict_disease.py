"""
Inference logic for the Disease Detection model.

Phase 1: Returns comprehensive disease advisory including symptoms,
treatment options (chemical/organic/systemic), prevention, causal agent,
and favorable conditions — everything a farmer needs.
"""
from __future__ import annotations

import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

from preprocessing.image_preprocessing import decode_and_resize_raw, to_batch
from utils.config_loader import ConfigLoader
from utils.exceptions import InferenceError
from utils.logger import get_logger
from utils.model_loader import ModelRegistry


def predict_disease(image_bytes: bytes, crop_type: str) -> dict:
    """
    Detect disease in a crop image and return comprehensive guidance.
    """
    logger = get_logger(__name__)
    try:
        settings = ConfigLoader.load("app_settings.json")
        treatment_kb = ConfigLoader.load("disease_treatment.json")["diseases"]
        img_size = tuple(settings["models"]["disease_detection"]["image_size"])

        model = ModelRegistry.get(f"disease_model_{crop_type}")
        labels = ModelRegistry.get_metadata(f"disease_labels_{crop_type}")
        if not labels:
            raise InferenceError(f"Disease labels for '{crop_type}' not loaded. Train the {crop_type} disease model first.")

        img = decode_and_resize_raw(image_bytes, target_size=img_size)
        img = preprocess_input(img)
        batch = to_batch(img)

        predictions = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(predictions))
        pred_label = labels[str(pred_idx)] if str(pred_idx) in labels else labels[pred_idx]
        confidence = float(predictions[pred_idx])

        kb_entry = treatment_kb.get(pred_label)

        if kb_entry is None:
            return {
                "crop": crop_type,
                "disease": pred_label,
                "display_name": pred_label.replace("_", " ").title(),
                "also_known_as": "",
                "confidence": round(confidence, 4),
                "severity": "unknown",
                "treatment": {"chemical": "N/A", "organic": "N/A", "systemic": "N/A"},
                "treatment_summary": "No treatment info available.",
                "prevention": "N/A",
                "symptoms": "N/A",
                "causal_agent": "Unknown",
                "favorable_conditions": "Unknown",
                "low_confidence_warning": True,
            }

        # Build structured treatment
        treatment_raw = kb_entry.get("treatment", {})
        if isinstance(treatment_raw, dict):
            treatment_dict = {
                "chemical": treatment_raw.get("chemical", "N/A"),
                "organic": treatment_raw.get("organic", "N/A"),
                "systemic": treatment_raw.get("systemic", "N/A"),
            }
            parts = [v for v in treatment_dict.values() if v != "N/A"]
            treatment_summary = " | ".join(parts)
        else:
            treatment_dict = {"chemical": str(treatment_raw), "organic": "N/A", "systemic": "N/A"}
            treatment_summary = str(treatment_raw)

        return {
            "crop": crop_type,
            "disease": pred_label,
            "display_name": kb_entry.get("display_name", pred_label.replace("_", " ").title()),
            "also_known_as": kb_entry.get("also_known_as", ""),
            "confidence": round(confidence, 4),
            "severity": kb_entry.get("default_severity", "unknown"),
            "treatment": treatment_dict,
            "treatment_summary": treatment_summary,
            "prevention": kb_entry.get("prevention", ""),
            "symptoms": kb_entry.get("symptoms", ""),
            "causal_agent": kb_entry.get("causal_agent", "Unknown"),
            "favorable_conditions": kb_entry.get("favorable_conditions", "Unknown"),
            "low_confidence_warning": confidence < settings.get("confidence_threshold", 0.5),
        }
    except InferenceError:
        raise
    except Exception as exc:
        logger.error(f"Disease prediction failed: {exc}")
        raise InferenceError(f"Disease prediction failed: {exc}") from exc
