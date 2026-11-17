"""
Model registry.

All ML models (sklearn joblib models + TensorFlow .h5 models) are loaded exactly
once, at FastAPI startup, and cached here. Inference modules pull models from this
registry instead of reloading from disk on every request.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from utils.config_loader import ConfigLoader, resolve_path
from utils.exceptions import ModelNotLoadedError
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Singleton-style in-memory store for all loaded models and metadata."""

    _models: Dict[str, Any] = {}
    _metadata: Dict[str, Any] = {}
    _loaded = False

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded

    @classmethod
    def get(cls, key: str) -> Any:
        if key not in cls._models:
            raise ModelNotLoadedError(
                f"Model '{key}' is not loaded. Did you run the training script "
                f"and start the app via main.py (which triggers load_all_models)?"
            )
        return cls._models[key]

    @classmethod
    def get_metadata(cls, key: str) -> Any:
        return cls._metadata.get(key)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        cls._models[key] = value

    @classmethod
    def set_metadata(cls, key: str, value: Any) -> None:
        cls._metadata[key] = value

    @classmethod
    def load_all_models(cls) -> None:
        """
        Load every model declared in config/app_settings.json.
        Missing model files are logged as warnings (not fatal) so the API can
        still start and serve endpoints whose models *are* trained -- useful
        during incremental development.
        """
        settings = ConfigLoader.load("app_settings.json")
        model_cfg = settings["models"]

        cls._load_crop_recommendation(model_cfg["crop_recommendation"])
        cls._load_stage_model(model_cfg["stage_classification"])
        cls._load_disease_model(model_cfg["disease_detection"])

        cls._loaded = True
        logger.info("Model registry initialization complete.")

    # ---- individual loaders -------------------------------------------------

    @classmethod
    def _load_crop_recommendation(cls, cfg: dict) -> None:
        model_path = resolve_path(cfg["path"])
        scaler_path = resolve_path(cfg["scaler_path"])
        encoder_path = resolve_path(cfg["label_encoder_path"])

        if model_path.exists():
            cls.set("crop_recommendation_model", joblib.load(model_path))
            logger.info(f"Loaded crop recommendation model from {model_path}")
        else:
            logger.warning(
                f"Crop recommendation model not found at {model_path}. "
                f"Run training/train_crop_model.py first."
            )

        if scaler_path.exists():
            cls.set("crop_recommendation_scaler", joblib.load(scaler_path))
        else:
            logger.warning(f"Crop scaler not found at {scaler_path}.")

        if encoder_path.exists():
            cls.set("crop_recommendation_encoder", joblib.load(encoder_path))
        else:
            logger.warning(f"Crop label encoder not found at {encoder_path}.")

    @classmethod
    def _load_stage_model(cls, cfg: dict) -> None:
        model_path = resolve_path(cfg["path"])
        labels_path = resolve_path(cfg["labels_path"])

        if model_path.exists():
            try:
                import tensorflow as tf  # imported lazily; heavy dependency

                cls.set("stage_model", tf.keras.models.load_model(
                    model_path, compile=False, safe_mode=False
                ))
                logger.info(f"Loaded stage classification model from {model_path}")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to load stage model: {exc}")
        else:
            logger.warning(
                f"Stage model not found at {model_path}. "
                f"Run training/train_stage_model.py first."
            )

        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                cls.set_metadata("stage_labels", json.load(f))
        else:
            logger.warning(f"Stage labels file not found at {labels_path}.")

    @classmethod
    def _load_disease_model(cls, cfg: dict) -> None:
        """Load per-crop disease models. Each crop has its own model + labels."""
        import tensorflow as tf

        per_crop = cfg.get("per_crop", {})
        if not per_crop:
            logger.warning("No per_crop disease models configured in app_settings.json.")
            return

        for crop, crop_cfg in per_crop.items():
            model_path = resolve_path(crop_cfg["path"])
            labels_path = resolve_path(crop_cfg["labels_path"])

            if model_path.exists():
                try:
                    model = tf.keras.models.load_model(
                        model_path, compile=False, safe_mode=False
                    )
                    cls.set(f"disease_model_{crop}", model)
                    logger.info(f"Loaded {crop} disease model from {model_path}")
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Failed to load {crop} disease model: {exc}")
            else:
                logger.warning(
                    f"{crop} disease model not found at {model_path}. "
                    f"Run: python training/train_disease_model.py --data-dir \"path/to/{crop}_data\""
                )

            if labels_path.exists():
                with open(labels_path, "r", encoding="utf-8") as f:
                    cls.set_metadata(f"disease_labels_{crop}", json.load(f))
            else:
                logger.warning(f"{crop} disease labels not found at {labels_path}.")
