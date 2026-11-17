"""
Config validation.

Runs at application startup (called from main.py lifespan) to validate every
JSON config file against a minimal schema. A typo in fertilizer_rules.json
previously would only surface as a cryptic KeyError when a real request hit
the fertilizer endpoint -- now it fails loudly at boot with a clear message
telling you exactly which file and key is wrong.

Validation is intentionally minimal: it checks required keys exist and have
the right types, not deep business logic. The goal is to catch "I edited the
JSON and introduced a typo" bugs, not to re-implement the full rule engine
in a schema.
"""
from __future__ import annotations

from typing import Any, Dict, List

from utils.config_loader import ConfigLoader
from utils.exceptions import ConfigError
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema definition
# Each schema is a dict of { key_path: expected_type }
# key_path uses dot notation for nested keys: "sensor_fallbacks.humidity.fallback_percent"
# ---------------------------------------------------------------------------

SCHEMAS: Dict[str, Dict[str, type]] = {
    "app_settings.json": {
        "models": dict,
        "models.crop_recommendation": dict,
        "models.crop_recommendation.path": str,
        "models.crop_recommendation.scaler_path": str,
        "models.crop_recommendation.label_encoder_path": str,
        "models.stage_classification": dict,
        "models.stage_classification.path": str,
        "models.stage_classification.image_size": list,
        "models.disease_detection": dict,
        "models.disease_detection.image_size": list,
        "supported_crops": list,
        "growth_stages": list,
        "sensor_fallbacks": dict,
        "sensor_fallbacks.humidity": dict,
        "sensor_fallbacks.humidity.fallback_percent": (int, float),
        "sensor_fallbacks.ph": dict,
        "sensor_fallbacks.ph.fallback_value": (int, float),
    },
    "fertilizer_rules.json": {
        "optimal_npk": dict,
        "optimal_npk.default": dict,
        "deficiency_thresholds": dict,
        "deficiency_thresholds.severe_percent": (int, float),
        "deficiency_thresholds.moderate_percent": (int, float),
        "deficiency_thresholds.mild_percent": (int, float),
        "nutrient_sources": dict,
        "nutrient_sources.N": dict,
        "nutrient_sources.P": dict,
        "nutrient_sources.K": dict,
        "messages": dict,
        "quantity_calculation": dict,
        "ph_rules": dict,
        "ph_rules.optimal_ph_range": dict,
    },
    "irrigation_rules.json": {
        "crop_stage_moisture_requirements": dict,
        "irrigation_amount_rules": dict,
        "irrigation_amount_rules.mm_per_percent_deficit": (int, float),
        "messages": dict,
    },
    "weather_rules.json": {
        "temperature_rules": list,
        "rainfall_rules": list,
    },
    "disease_treatment.json": {
        "diseases": dict,
    },
    "pest_rules.json": {
        "pests": dict,
        "general_guidelines": dict,
    },
}


def _resolve_path(data: Any, key_path: str) -> Any:
    """Walk dot-separated key_path into nested dicts. Returns None if any level is missing."""
    parts = key_path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


_MISSING = object()


def _validate_one(filename: str, schema: Dict[str, Any]) -> List[str]:
    errors = []
    try:
        data = ConfigLoader.load(filename)
    except FileNotFoundError:
        return [f"Config file not found: {filename}"]
    except Exception as exc:  # noqa: BLE001
        return [f"Could not parse {filename}: {exc}"]

    for key_path, expected_type in schema.items():
        value = _resolve_path(data, key_path)
        if value is _MISSING:
            errors.append(f"[{filename}] Missing required key: '{key_path}'")
            continue
        # expected_type may be a tuple (int, float) for numeric flexibility
        if not isinstance(value, expected_type):
            errors.append(
                f"[{filename}] Key '{key_path}' should be {expected_type}, "
                f"got {type(value).__name__} (value: {value!r})"
            )
    return errors


def validate_all_configs() -> None:
    """
    Validate all config files. Raises ConfigError if any file has problems.
    Call this once from the FastAPI lifespan startup so problems are caught
    before any request is served.
    """
    all_errors: List[str] = []
    for filename, schema in SCHEMAS.items():
        errors = _validate_one(filename, schema)
        all_errors.extend(errors)

    if all_errors:
        error_text = "\n  ".join(all_errors)
        raise ConfigError(
            f"Config validation failed with {len(all_errors)} error(s):\n  {error_text}\n"
            f"Fix the JSON files listed above, then restart the application."
        )

    logger.info(f"All {len(SCHEMAS)} config files passed validation.")