"""
Fertilizer Recommendation Service.

Entirely rule-driven from config/fertilizer_rules.json -- no thresholds or
crop-specific numbers are hardcoded here. To change behavior, edit the JSON
config; this module only implements the calculation logic.
"""
from __future__ import annotations

from typing import Dict, List

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


def _classify_severity(deficit_percent: float, thresholds: dict) -> str | None:
    if deficit_percent <= 0:
        return None
    if deficit_percent >= thresholds["severe_percent"]:
        return "severe"
    if deficit_percent >= thresholds["moderate_percent"]:
        return "moderate"
    if deficit_percent >= thresholds["mild_percent"]:
        return "mild"
    return None


def _check_ph(crop: str, ph: float | None, rules: dict) -> tuple[list[str], dict | None]:
    """Evaluate soil pH against the crop's optimal range and recommend lime/sulfur if needed."""
    if ph is None:
        return [], None

    ph_rules = rules.get("ph_rules")
    if not ph_rules:
        return [], None

    optimal = ph_rules["optimal_ph_range"].get(crop, ph_rules["optimal_ph_range"]["default"])
    messages = ph_rules["messages"]
    min_ph, max_ph = optimal["min"], optimal["max"]

    if ph < min_ph:
        msg = messages["too_acidic"].format(crop=crop, ph=ph, min=min_ph, max=max_ph)
        detail = {"ph": ph, "status": "too_acidic", "optimal_range": optimal, "recommended_action": "Apply agricultural lime"}
    elif ph > max_ph:
        msg = messages["too_alkaline"].format(crop=crop, ph=ph, min=min_ph, max=max_ph)
        detail = {"ph": ph, "status": "too_alkaline", "optimal_range": optimal, "recommended_action": "Apply elemental sulfur or organic matter"}
    else:
        msg = messages["optimal"].format(crop=crop, ph=ph, min=min_ph, max=max_ph)
        detail = {"ph": ph, "status": "optimal", "optimal_range": optimal}

    return [msg], detail


def recommend_fertilizer(crop: str, n: float, p: float, k: float, ph: float | None = None) -> dict:
    """
    Compare current N/P/K readings against the optimal levels configured for
    the given crop and recommend specific fertilizers + quantities.

    `ph` is optional -- pass it when available (real sensor reading or the
    configured default) to also get lime/sulfur guidance based on soil pH,
    since pH affects how available N/P/K already are to the plant.
    """
    rules = ConfigLoader.load("fertilizer_rules.json")
    optimal = rules["optimal_npk"].get(crop, rules["optimal_npk"]["default"])
    thresholds = rules["deficiency_thresholds"]
    sources = rules["nutrient_sources"]
    messages = rules["messages"]
    quantity_rules = rules["quantity_calculation"]

    current_values = {"N": n, "P": p, "K": k}
    recommendations: List[str] = []
    details: List[Dict] = []
    any_deficiency = False

    for nutrient, current in current_values.items():
        opt = optimal[nutrient]
        deficit = opt - current

        if deficit <= 0:
            # Excess check
            excess_percent = ((current - opt) / opt) * 100 if opt else 0
            if excess_percent >= thresholds["mild_percent"]:
                recommendations.append(
                    messages["excess"].format(nutrient=nutrient, crop=crop, current=current, optimal=opt)
                )
                details.append({
                    "nutrient": nutrient, "status": "excess",
                    "current": current, "optimal": opt,
                })
            continue

        deficit_percent = (deficit / opt) * 100 if opt else 0
        severity = _classify_severity(deficit_percent, thresholds)

        if severity is None:
            continue

        any_deficiency = True
        source_info = sources[nutrient]
        nutrient_content_pct = source_info["chemical_nutrient_content_percent"]
        quantity_kg_per_ha = (deficit / nutrient_content_pct) * 100
        quantity_kg_per_ha = min(quantity_kg_per_ha, quantity_rules["safety_cap_kg_per_ha"])

        basis = f"based on a {deficit_percent:.1f}% deficit vs optimal {opt} for {crop}"
        recommendations.append(
            messages["deficiency"].format(
                crop=crop, severity=severity, nutrient=nutrient,
                current=current, optimal=opt,
                quantity_kg_per_ha=round(quantity_kg_per_ha, 1),
                source=source_info["chemical"], basis=basis,
            )
        )
        details.append({
            "nutrient": nutrient, "status": "deficient", "severity": severity,
            "current": current, "optimal": opt,
            "recommended_fertilizer": source_info["chemical"],
            "organic_alternative": source_info["organic_alternative"],
            "quantity_kg_per_ha": round(quantity_kg_per_ha, 1),
        })

    if not any_deficiency and not recommendations:
        recommendations.append(messages["no_deficiency"].format(crop=crop))

    ph_messages, ph_detail = _check_ph(crop, ph, rules)
    recommendations.extend(ph_messages)
    if ph_detail:
        details.append({"nutrient": "ph", **ph_detail})

    return {
        "crop": crop,
        "npk_input": current_values,
        "optimal_npk": optimal,
        "recommendations": recommendations,
        "details": details,
    }
