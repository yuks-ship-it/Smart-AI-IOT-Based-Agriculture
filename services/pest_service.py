"""
Pest Recommendation Service.

Given a detected pest class name and the current crop growth stage, returns
specific management advice including chemical, organic, and cultural controls,
with stage-specific spray restrictions applied automatically.

This is rule-driven from config/pest_rules.json -- no hardcoded pest names
or treatment text in this file.
"""
from __future__ import annotations

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


def get_pest_advice(pest_class: str, crop: str, stage: str) -> dict:
    """
    Look up management advice for a detected pest at a specific crop stage.

    Args:
        pest_class: folder name used during training, e.g. "rice_yellow_stem_borer"
        crop: rice / wheat / maize
        stage: seedling / tillering / vegetative / flowering / maturity

    Returns:
        Full pest management dict with chemical, organic, cultural advice
        and any spray restrictions for the current stage.
    """
    rules = ConfigLoader.load("pest_rules.json")
    pests = rules["pests"]
    restrictions = rules.get("stage_spray_restrictions", {}).get("restrictions", [])
    guidelines = rules.get("general_guidelines", {})

    pest_info = pests.get(pest_class)
    if not pest_info:
        logger.warning(
            f"Pest class '{pest_class}' not found in pest_rules.json. "
            f"Add an entry for it to get treatment advice."
        )
        return {
            "pest_class": pest_class,
            "display_name": pest_class.replace("_", " ").title(),
            "crop": crop,
            "stage": stage,
            "severity": "unknown",
            "advice": "No treatment information available for this pest. "
                      "Add an entry to config/pest_rules.json.",
            "chemical_control": None,
            "organic_control": None,
            "stage_specific_advice": None,
            "spray_restrictions": [],
            "economic_threshold": None,
        }

    # Get stage-specific management advice if available
    stage_advice = pest_info.get("stage_specific_advice", {}).get(stage)

    # Get management for this stage
    management = pest_info.get("management", {})
    chemical = management.get("chemical")
    if isinstance(chemical, dict):
        # Some pests have different chemical advice per stage
        chemical = chemical.get(stage) or chemical.get("vegetative") or str(chemical)

    # Apply stage spray restrictions
    active_restrictions = [
        r["restriction"]
        for r in restrictions
        if r["stage"] == stage
    ]

    # Build human-readable summary paragraph
    summary_parts = []
    display = pest_info.get("display_name", pest_class)
    severity = pest_info.get("default_severity", "moderate")

    summary_parts.append(
        f"{display} detected on {crop} at {stage} stage (severity: {severity})."
    )

    threshold = pest_info.get("economic_threshold")
    if threshold:
        summary_parts.append(f"Economic threshold: {threshold}")

    if stage_advice:
        summary_parts.append(f"Recommended action at {stage} stage: {stage_advice}")
    elif chemical:
        summary_parts.append(f"Chemical control: {chemical}")

    organic = management.get("organic")
    if organic:
        summary_parts.append(f"Organic alternative: {organic}")

    if active_restrictions:
        summary_parts.append(
            f"IMPORTANT RESTRICTION at {stage} stage: {' '.join(active_restrictions)}"
        )

    return {
        "pest_class": pest_class,
        "display_name": display,
        "scientific_name": pest_info.get("scientific_name"),
        "crop": crop,
        "stage": stage,
        "severity": severity,
        "damage_symptoms": pest_info.get("damage_symptoms"),
        "economic_threshold": threshold,
        "chemical_control": chemical,
        "organic_control": organic,
        "cultural_control": management.get("cultural"),
        "stage_specific_advice": stage_advice,
        "spray_restrictions": active_restrictions,
        "prevention": pest_info.get("prevention"),
        "summary": " ".join(summary_parts),
        "general_guidelines": guidelines,
    }


def list_pests_for_crop(crop: str) -> list[dict]:
    """Return all known pests for a given crop from the config."""
    rules = ConfigLoader.load("pest_rules.json")
    return [
        {
            "pest_class": key,
            "display_name": v.get("display_name"),
            "scientific_name": v.get("scientific_name"),
            "severity": v.get("default_severity"),
        }
        for key, v in rules["pests"].items()
        if v.get("crop") == crop
    ]