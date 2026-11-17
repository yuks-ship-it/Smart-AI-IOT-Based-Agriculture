"""
Advisory Engine — Phase 1 (Rice Disease Focused).

Produces a detailed, farmer-friendly advisory that:
- Gives full disease diagnosis with symptoms, cause, and why it happened
- Provides treatment options: chemical, organic, systemic
- Resolves conflicts (rain vs spray, irrigation vs drainage)
- Prioritizes actions in the correct order
- Includes prevention for next season
"""
from __future__ import annotations

from typing import Optional

from inference.predict_disease import predict_disease
from inference.predict_stage import predict_stage
from services.fertilizer_service import recommend_fertilizer
from services.irrigation_service import recommend_irrigation
from services.weather_service import get_weather_advisory
from services.weather_fetch_service import get_ph_default
from utils.logger import get_logger

logger = get_logger(__name__)


def _detect_conflicts(disease_info: dict | None, weather_info: dict, irrigation_info: dict) -> list[dict]:
    """
    Detect conflicts between recommendations and return resolution advice.
    """
    conflicts = []
    rain_actions = weather_info.get("recommended_actions", [])
    has_rain = "delay_pesticide" in rain_actions or "delay_fertilizer" in rain_actions

    # Conflict 1: Disease needs spray but rain is expected
    if has_rain and disease_info and disease_info.get("disease", "healthy") != "healthy":
        severity = disease_info.get("severity", "unknown")
        if severity == "high":
            conflicts.append({
                "type": "rain_vs_spray",
                "severity": "critical",
                "message": (
                    "Rain is expected but a HIGH-severity disease requires immediate treatment. "
                    "Spray in the next dry window (early morning before rain). "
                    "Use a sticker/adjuvant to improve spray adhesion."
                ),
            })
        else:
            conflicts.append({
                "type": "rain_vs_spray",
                "severity": "moderate",
                "message": (
                    "Rain is expected. Delay fungicide spray until after rain to avoid washoff. "
                    "Monitor disease closely — if it spreads, spray at next dry opportunity."
                ),
            })

    # Conflict 2: Disease says drain but irrigation says water needed
    if disease_info and disease_info.get("disease", "healthy") != "healthy":
        treatment = disease_info.get("treatment", {})
        systemic = treatment.get("systemic", "") if isinstance(treatment, dict) else ""
        if "drain" in systemic.lower() and irrigation_info.get("irrigation_required"):
            conflicts.append({
                "type": "drain_vs_irrigate",
                "severity": "critical",
                "message": (
                    "Disease treatment requires DRAINING the field, but soil moisture is low. "
                    "PRIORITY: Drain to stop disease spread first. Resume irrigation only "
                    "after disease is controlled (7-10 days)."
                ),
            })

    # Conflict 3: Fertilizer recommended but rain expected
    if has_rain:
        conflicts.append({
            "type": "rain_vs_fertilizer",
            "severity": "moderate",
            "message": "Delay fertilizer application until after rain to prevent runoff.",
        })

    return conflicts


def _build_advisory_text(
    crop: str,
    crop_stage: str | None,
    disease_info: dict | None,
    fertilizer_info: dict,
    irrigation_info: dict,
    weather_info: dict,
    conflicts: list[dict],
) -> str:
    """Build the final farmer-friendly advisory text."""
    lines = []

    # Header
    stage_text = f" ({crop_stage.title()} Stage)" if crop_stage else ""
    lines.append(f"SMART AGRICULTURE ADVISORY - {crop.upper()}{stage_text}")
    lines.append("=" * 55)

    # === DISEASE SECTION ===
    if disease_info and disease_info.get("disease", "healthy") != "healthy":
        d = disease_info
        lines.append("")
        lines.append(f"[DISEASE ALERT] {d.get('display_name', d['disease'])}")
        if d.get("also_known_as"):
            lines.append(f"  Also known as: {d['also_known_as']}")
        lines.append(f"  Severity: {d.get('severity', 'unknown').upper()}")
        lines.append(f"  Confidence: {d.get('confidence', 0)*100:.1f}%")
        lines.append(f"  Caused by: {d.get('causal_agent', 'Unknown')}")
        lines.append("")

        if d.get("symptoms"):
            lines.append(f"  Symptoms: {d['symptoms']}")
            lines.append("")

        if d.get("favorable_conditions"):
            lines.append(f"  Why it happened: {d['favorable_conditions']}")
            lines.append("")

        # Treatment options
        treatment = d.get("treatment", {})
        if isinstance(treatment, dict):
            lines.append("  TREATMENT OPTIONS:")
            if treatment.get("chemical") and treatment["chemical"] != "N/A":
                lines.append(f"    Chemical: {treatment['chemical']}")
            if treatment.get("organic") and treatment["organic"] != "N/A":
                lines.append(f"    Organic:  {treatment['organic']}")
            if treatment.get("systemic") and treatment["systemic"] != "N/A":
                lines.append(f"    Systemic: {treatment['systemic']}")
        lines.append("")

        if d.get("prevention"):
            lines.append(f"  Prevention (next season): {d['prevention']}")
            lines.append("")

    elif disease_info and disease_info.get("disease") == "healthy":
        lines.append("")
        lines.append("[HEALTH STATUS] No disease detected. Your crop looks healthy.")
        lines.append("")

    # === CONFLICTS ===
    critical_conflicts = [c for c in conflicts if c["severity"] == "critical"]
    if critical_conflicts:
        lines.append("[IMPORTANT CONFLICTS]")
        for c in critical_conflicts:
            lines.append(f"  >> {c['message']}")
        lines.append("")

    # === IRRIGATION ===
    if irrigation_info.get("irrigation_required"):
        drain_conflict = any(c["type"] == "drain_vs_irrigate" for c in conflicts)
        if drain_conflict:
            lines.append("[IRRIGATION] HOLD - Drain field first for disease control.")
        else:
            rain_conflict = any(c["type"] == "rain_vs_fertilizer" for c in conflicts)
            if rain_conflict:
                lines.append(f"[IRRIGATION] Needed: {irrigation_info['recommended_amount_mm']} mm")
                lines.append("  >> Wait for rain first. Check moisture after rainfall.")
            else:
                lines.append(f"[IRRIGATION] {irrigation_info['message']}")
    elif irrigation_info.get("message"):
        lines.append(f"[IRRIGATION] {irrigation_info['message']}")
    lines.append("")

    # === FERTILIZER ===
    fert_details = fertilizer_info.get("details", [])
    deficiencies = [d for d in fert_details if d.get("status") == "deficient"]
    excesses = [d for d in fert_details if d.get("status") == "excess"]
    ph_issues = [d for d in fert_details if d.get("nutrient") == "ph" and d.get("status") in ("too_acidic", "too_alkaline")]

    rain_delay = any(c["type"] == "rain_vs_fertilizer" for c in conflicts)

    if deficiencies or excesses or ph_issues:
        lines.append("[FERTILIZER]")
        if rain_delay:
            lines.append("  >> DELAY: Rain expected - apply after dry conditions return.")
        for d in deficiencies:
            lines.append(
                f"  {d['nutrient']} deficiency ({d['severity']}): "
                f"Apply {d['quantity_kg_per_ha']} kg/ha of {d['recommended_fertilizer']}"
            )
        for d in excesses:
            lines.append(f"  {d['nutrient']}: EXCESSIVE - stop further application.")
        for d in ph_issues:
            lines.append(f"  Soil pH: {d['ph']} ({d['status'].replace('_', ' ')}) - {d.get('recommended_action', 'Correct pH')}")
        lines.append("")
    else:
        lines.append("[FERTILIZER] NPK levels are adequate. No action needed.")
        lines.append("")

    # === WEATHER ===
    advisories = weather_info.get("advisories", [])
    if advisories:
        lines.append("[WEATHER]")
        for adv in advisories:
            lines.append(f"  {adv}")
        lines.append("")

    # === PRIORITY ACTION LIST ===
    actions = _build_action_list(disease_info, fertilizer_info, irrigation_info, weather_info, conflicts)
    if actions:
        lines.append("=" * 55)
        lines.append("PRIORITY ACTION LIST (do in this order):")
        for i, action in enumerate(actions, 1):
            lines.append(f"  {i}. {action}")
        lines.append("=" * 55)

    return "\n".join(lines)


def _build_action_list(disease_info, fertilizer_info, irrigation_info, weather_info, conflicts) -> list[str]:
    """Build ordered action list considering all conflicts."""
    actions = []

    has_disease = disease_info and disease_info.get("disease", "healthy") != "healthy"
    has_rain = "delay_pesticide" in weather_info.get("recommended_actions", [])
    needs_drain = False

    # Disease systemic actions first (drain, stop N)
    if has_disease:
        treatment = disease_info.get("treatment", {})
        systemic = treatment.get("systemic", "") if isinstance(treatment, dict) else ""
        if "drain" in systemic.lower():
            actions.append("DRAIN the field immediately (stops disease spread)")
            needs_drain = True
        if "stop" in systemic.lower() and "nitrogen" in systemic.lower():
            actions.append("Stop nitrogen fertilizer application")

    # Disease spray
    if has_disease:
        treatment = disease_info.get("treatment", {})
        chemical = treatment.get("chemical", "") if isinstance(treatment, dict) else ""
        if chemical and chemical != "N/A":
            short_chem = chemical[:80] + "..." if len(chemical) > 80 else chemical
            if has_rain:
                actions.append(f"Wait for dry window then spray: {short_chem}")
            else:
                actions.append(f"Spray now: {short_chem}")

    # Irrigation (if not conflicting with drain)
    if irrigation_info.get("irrigation_required") and not needs_drain:
        if has_rain:
            actions.append("After rain: Check soil moisture, irrigate if still below threshold")
        else:
            actions.append(f"Irrigate: {irrigation_info['recommended_amount_mm']} mm of water")

    # Fertilizer
    deficiencies = [d for d in fertilizer_info.get("details", []) if d.get("status") == "deficient"]
    if deficiencies:
        fert_text = ", ".join(f"{d['quantity_kg_per_ha']}kg/ha {d['recommended_fertilizer']}" for d in deficiencies)
        if has_rain:
            actions.append(f"After rain: Apply fertilizer ({fert_text})")
        else:
            actions.append(f"Apply fertilizer: {fert_text}")

    if not actions:
        actions.append("Continue regular monitoring. No immediate action needed.")

    return actions


def generate_advisory(
    crop: str,
    n: float,
    p: float,
    k: float,
    soil_moisture: float,
    temperature: float,
    humidity: float,
    ph: Optional[float] = None,
    rainfall_probability: float = 0.0,
    rainfall_mm: float = 0.0,
    stage_image_bytes: Optional[bytes] = None,
    disease_image_bytes: Optional[bytes] = None,
    known_stage: Optional[str] = None,
) -> dict:
    """
    Run every applicable AI module and combine into one advisory.
    """
    stage_result = None
    disease_result = None
    resolved_ph = ph if ph is not None else get_ph_default()

    if stage_image_bytes is not None:
        try:
            stage_result = predict_stage(stage_image_bytes, crop)
        except Exception as exc:
            logger.warning(f"Stage prediction skipped: {exc}")

    if disease_image_bytes is not None:
        try:
            disease_result = predict_disease(disease_image_bytes, crop)
        except Exception as exc:
            logger.warning(f"Disease prediction skipped: {exc}")

    resolved_stage = (
        stage_result["growth_stage"] if stage_result else (known_stage or "vegetative")
    )

    fertilizer_result = recommend_fertilizer(crop, n, p, k, ph=resolved_ph)
    irrigation_result = recommend_irrigation(crop, resolved_stage, soil_moisture)
    weather_result = get_weather_advisory(temperature, rainfall_probability, rainfall_mm)

    conflicts = _detect_conflicts(disease_result, weather_result, irrigation_result)

    final_text = _build_advisory_text(
        crop=crop,
        crop_stage=stage_result["growth_stage"] if stage_result else known_stage,
        disease_info=disease_result,
        fertilizer_info=fertilizer_result,
        irrigation_info=irrigation_result,
        weather_info=weather_result,
        conflicts=conflicts,
    )

    return {
        "crop": crop,
        "crop_stage": stage_result,
        "disease": disease_result,
        "fertilizer": fertilizer_result,
        "irrigation": irrigation_result,
        "weather_advice": weather_result,
        "conflicts": conflicts,
        "final_advisory": final_text,
    }
