"""
Irrigation Recommendation Service.

Entirely rule-driven from config/irrigation_rules.json.
"""
from __future__ import annotations

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


def recommend_irrigation(crop: str, stage: str, soil_moisture: float) -> dict:
    """
    Determine whether irrigation is needed based on current soil moisture,
    crop type, and growth stage, and how much water to apply if so.
    """
    rules = ConfigLoader.load("irrigation_rules.json")
    crop_rules = rules["crop_stage_moisture_requirements"].get(crop)

    if crop_rules and stage in crop_rules:
        req = crop_rules[stage]
    else:
        defaults = rules["default_moisture_thresholds"]
        req = {"min_moisture": defaults["low"], "ideal_moisture": defaults["medium"]}
        logger.warning(
            f"No specific irrigation rule for crop='{crop}' stage='{stage}'. "
            f"Falling back to default thresholds."
        )

    min_moisture = req["min_moisture"]
    ideal_moisture = req["ideal_moisture"]
    amount_rules = rules["irrigation_amount_rules"]
    messages = rules["messages"]

    if soil_moisture < min_moisture:
        deficit_percent = min_moisture - soil_moisture
        amount_mm = deficit_percent * amount_rules["mm_per_percent_deficit"]
        amount_mm = max(amount_rules["min_recommended_mm"], min(amount_mm, amount_rules["max_recommended_mm"]))
        irrigation_required = True
        message = messages["irrigation_required"].format(
            crop=crop, stage=stage, moisture=soil_moisture,
            min_moisture=min_moisture, amount_mm=round(amount_mm, 1),
        )
    elif soil_moisture < ideal_moisture:
        deficit_percent = ideal_moisture - soil_moisture
        amount_mm = deficit_percent * amount_rules["mm_per_percent_deficit"] * 0.5
        amount_mm = max(amount_rules["min_recommended_mm"], min(amount_mm, amount_rules["max_recommended_mm"]))
        irrigation_required = False
        message = messages["irrigation_optional"].format(
            crop=crop, stage=stage, moisture=soil_moisture,
            ideal_moisture=ideal_moisture, amount_mm=round(amount_mm, 1),
        )
    else:
        amount_mm = 0
        irrigation_required = False
        message = messages["no_irrigation_needed"].format(crop=crop, stage=stage, moisture=soil_moisture)

    return {
        "crop": crop,
        "stage": stage,
        "soil_moisture": soil_moisture,
        "min_moisture_required": min_moisture,
        "ideal_moisture": ideal_moisture,
        "irrigation_required": irrigation_required,
        "recommended_amount_mm": round(amount_mm, 1),
        "message": message,
    }
