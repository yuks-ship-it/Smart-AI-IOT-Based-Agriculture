"""
Demo: Rice Advisory System - Phase 1.

Simulates what the advisory system outputs for a rice farmer
using the rule-based services (no trained model needed for this demo).

Run: python demo_advisory.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from services.fertilizer_service import recommend_fertilizer
from services.irrigation_service import recommend_irrigation
from services.weather_service import get_weather_advisory


def load_disease_kb():
    """Load disease knowledge base directly for demo."""
    with open("config/disease_treatment.json", "r", encoding="utf-8") as f:
        return json.load(f)["diseases"]


def simulate_disease_detection(disease_class: str) -> dict:
    """Simulate what predict_disease() returns without needing the model."""
    kb = load_disease_kb()
    entry = kb.get(disease_class, {})
    treatment_raw = entry.get("treatment", {})
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
        "crop": "rice",
        "disease": disease_class,
        "display_name": entry.get("display_name", disease_class),
        "also_known_as": entry.get("also_known_as", ""),
        "confidence": 0.87,
        "severity": entry.get("default_severity", "unknown"),
        "treatment": treatment_dict,
        "treatment_summary": treatment_summary,
        "prevention": entry.get("prevention", ""),
        "symptoms": entry.get("symptoms", ""),
        "causal_agent": entry.get("causal_agent", "Unknown"),
        "favorable_conditions": entry.get("favorable_conditions", "Unknown"),
        "low_confidence_warning": False,
    }


def build_demo_advisory(crop, stage, disease_info, fertilizer_info, irrigation_info, weather_info):
    """Build advisory text for demo output."""
    lines = []
    lines.append(f"SMART AGRICULTURE ADVISORY - {crop.upper()} ({stage.title()} Stage)")
    lines.append("=" * 55)

    # Disease section
    d = disease_info
    if d["disease"] != "healthy":
        lines.append("")
        lines.append(f"[DISEASE ALERT] {d['display_name']}")
        if d.get("also_known_as"):
            lines.append(f"  Also known as: {d['also_known_as']}")
        lines.append(f"  Severity: {d['severity'].upper()}")
        lines.append(f"  Confidence: {d['confidence']*100:.1f}%")
        lines.append(f"  Caused by: {d['causal_agent']}")
        lines.append("")
        lines.append(f"  Symptoms: {d['symptoms']}")
        lines.append("")
        lines.append(f"  Why it happened: {d['favorable_conditions']}")
        lines.append("")
        lines.append("  TREATMENT OPTIONS:")
        t = d["treatment"]
        if t.get("chemical") and t["chemical"] != "N/A":
            lines.append(f"    Chemical: {t['chemical']}")
        if t.get("organic") and t["organic"] != "N/A":
            lines.append(f"    Organic:  {t['organic']}")
        if t.get("systemic") and t["systemic"] != "N/A":
            lines.append(f"    Systemic: {t['systemic']}")
        lines.append("")
        lines.append(f"  Prevention (next season): {d['prevention']}")
    else:
        lines.append("")
        lines.append("[HEALTH STATUS] No disease detected. Your crop looks healthy.")
        lines.append(f"  {d['prevention']}")

    # Conflicts
    rain_actions = weather_info.get("recommended_actions", [])
    has_rain = "delay_pesticide" in rain_actions
    if has_rain and d["disease"] != "healthy" and d["severity"] == "high":
        lines.append("")
        lines.append("[IMPORTANT CONFLICT]")
        lines.append("  >> Rain expected but HIGH-severity disease detected!")
        lines.append("  >> Spray in next dry window (early morning before rain).")
        lines.append("  >> Use sticker/adjuvant for better adhesion.")

    # Check drain conflict
    needs_drain = False
    systemic = d["treatment"].get("systemic", "") if d["disease"] != "healthy" else ""
    if "drain" in systemic.lower() and irrigation_info.get("irrigation_required"):
        lines.append("")
        lines.append("[CONFLICT] Disease requires DRAINING but soil is dry!")
        lines.append("  >> PRIORITY: Drain first to stop spread.")
        lines.append("  >> Resume irrigation after 7-10 days when disease controlled.")
        needs_drain = True

    # Irrigation
    lines.append("")
    if irrigation_info.get("irrigation_required"):
        if needs_drain:
            lines.append("[IRRIGATION] HOLD - Drain field first for disease control.")
        elif has_rain:
            lines.append(f"[IRRIGATION] Needed: {irrigation_info['recommended_amount_mm']} mm")
            lines.append("  >> Wait for rain. Check moisture again after rainfall.")
        else:
            lines.append(f"[IRRIGATION] {irrigation_info['message']}")
    else:
        lines.append(f"[IRRIGATION] {irrigation_info['message']}")

    # Fertilizer
    lines.append("")
    fert_details = fertilizer_info.get("details", [])
    deficiencies = [x for x in fert_details if x.get("status") == "deficient"]
    ph_issues = [x for x in fert_details if x.get("nutrient") == "ph" and x.get("status") in ("too_acidic", "too_alkaline")]

    if deficiencies or ph_issues:
        lines.append("[FERTILIZER]")
        if has_rain:
            lines.append("  >> DELAY: Rain expected - apply after dry conditions.")
        for det in deficiencies:
            lines.append(
                f"  {det['nutrient']} ({det['severity']}): "
                f"{det['quantity_kg_per_ha']} kg/ha of {det['recommended_fertilizer']}"
            )
        for det in ph_issues:
            lines.append(f"  pH={det['ph']} ({det['status'].replace('_', ' ')}): {det.get('recommended_action', 'Correct pH')}")
    else:
        lines.append("[FERTILIZER] NPK levels adequate. No action needed.")

    # Weather
    lines.append("")
    lines.append("[WEATHER]")
    for adv in weather_info.get("advisories", []):
        lines.append(f"  {adv}")

    # Priority actions
    lines.append("")
    lines.append("=" * 55)
    lines.append("PRIORITY ACTION LIST (do in this order):")
    action_num = 1
    if needs_drain:
        lines.append(f"  {action_num}. DRAIN the field immediately (stops disease spread)")
        action_num += 1
    if "stop" in systemic.lower() and "nitrogen" in systemic.lower():
        lines.append(f"  {action_num}. Stop nitrogen fertilizer application")
        action_num += 1
    if d["disease"] != "healthy":
        chemical = d["treatment"].get("chemical", "")
        if chemical and chemical != "N/A":
            short = chemical[:70] + "..." if len(chemical) > 70 else chemical
            prefix = "Wait for dry window then spray" if has_rain else "Spray now"
            lines.append(f"  {action_num}. {prefix}: {short}")
            action_num += 1
    if irrigation_info.get("irrigation_required") and not needs_drain:
        prefix = "After rain: Check moisture, irrigate if needed" if has_rain else f"Irrigate: {irrigation_info['recommended_amount_mm']} mm"
        lines.append(f"  {action_num}. {prefix}")
        action_num += 1
    if deficiencies:
        fert_text = ", ".join(f"{x['quantity_kg_per_ha']}kg/ha {x['recommended_fertilizer']}" for x in deficiencies)
        prefix = "After rain: Apply" if has_rain else "Apply"
        lines.append(f"  {action_num}. {prefix} fertilizer: {fert_text}")
        action_num += 1
    lines.append("=" * 55)

    return "\n".join(lines)


def run_demo():
    print()
    print("#" * 60)
    print("##  SMART AGRICULTURE ADVISORY SYSTEM - DEMO")
    print("##  Phase 1: Rice Disease Advisory")
    print("#" * 60)

    # =========================================================
    # SCENARIO 1: Bacterial Leaf Blight + Rain + Low moisture
    # =========================================================
    print("\n" + "=" * 60)
    print("  SCENARIO 1: Bacterial Leaf Blight (worst case)")
    print("  Rice, Flowering stage, rain expected, low moisture")
    print("=" * 60)

    crop = "rice"
    stage = "flowering"
    n, p, k = 65, 28, 50
    soil_moisture = 48
    temperature = 31
    ph = 5.3
    rainfall_prob = 65
    rainfall_mm = 20

    print(f"\n  SENSOR INPUT:")
    print(f"    Crop: {crop} | Stage: {stage}")
    print(f"    NPK: N={n}, P={p}, K={k} | pH={ph}")
    print(f"    Soil Moisture: {soil_moisture}%")
    print(f"    Temperature: {temperature} C | Rain: {rainfall_prob}% ({rainfall_mm}mm)")
    print(f"    Disease Image: bacterial_leaf_blight (87% confidence)")

    disease = simulate_disease_detection("bacterial_leaf_blight")
    fertilizer = recommend_fertilizer(crop, n, p, k, ph=ph)
    irrigation = recommend_irrigation(crop, stage, soil_moisture)
    weather = get_weather_advisory(temperature, rainfall_prob, rainfall_mm)

    advisory_text = build_demo_advisory(crop, stage, disease, fertilizer, irrigation, weather)
    print(f"\n  {'─' * 55}")
    print(f"  ADVISORY OUTPUT:")
    print(f"  {'─' * 55}")
    for line in advisory_text.split("\n"):
        print(f"  {line}")

    # =========================================================
    # SCENARIO 2: Leaf Blast + Normal conditions
    # =========================================================
    print("\n\n" + "=" * 60)
    print("  SCENARIO 2: Leaf Blast (moderate, no rain)")
    print("  Rice, Tillering stage, normal weather")
    print("=" * 60)

    stage2 = "tillering"
    soil_moisture2 = 52
    temperature2 = 26
    rainfall_prob2 = 10
    rainfall_mm2 = 0

    print(f"\n  SENSOR INPUT:")
    print(f"    Crop: {crop} | Stage: {stage2}")
    print(f"    NPK: N=80, P=45, K=48 | pH=6.2")
    print(f"    Soil Moisture: {soil_moisture2}%")
    print(f"    Temperature: {temperature2} C | Rain: {rainfall_prob2}%")
    print(f"    Disease Image: leaf_blast (92% confidence)")

    disease2 = simulate_disease_detection("leaf_blast")
    disease2["confidence"] = 0.92
    fertilizer2 = recommend_fertilizer(crop, 80, 45, 48, ph=6.2)
    irrigation2 = recommend_irrigation(crop, stage2, soil_moisture2)
    weather2 = get_weather_advisory(temperature2, rainfall_prob2, rainfall_mm2)

    advisory2 = build_demo_advisory(crop, stage2, disease2, fertilizer2, irrigation2, weather2)
    print(f"\n  {'─' * 55}")
    print(f"  ADVISORY OUTPUT:")
    print(f"  {'─' * 55}")
    for line in advisory2.split("\n"):
        print(f"  {line}")

    # =========================================================
    # SCENARIO 3: Healthy crop
    # =========================================================
    print("\n\n" + "=" * 60)
    print("  SCENARIO 3: Healthy Crop")
    print("  Rice, Vegetative stage, everything normal")
    print("=" * 60)

    stage3 = "vegetative"
    print(f"\n  SENSOR INPUT:")
    print(f"    Crop: {crop} | Stage: {stage3}")
    print(f"    NPK: N=95, P=48, K=50 | pH=6.0")
    print(f"    Soil Moisture: 58%")
    print(f"    Temperature: 28 C | Rain: 15%")
    print(f"    Disease Image: healthy (94% confidence)")

    disease3 = simulate_disease_detection("healthy")
    disease3["confidence"] = 0.94
    fertilizer3 = recommend_fertilizer(crop, 95, 48, 50, ph=6.0)
    irrigation3 = recommend_irrigation(crop, stage3, 58)
    weather3 = get_weather_advisory(28, 15, 0)

    advisory3 = build_demo_advisory(crop, stage3, disease3, fertilizer3, irrigation3, weather3)
    print(f"\n  {'─' * 55}")
    print(f"  ADVISORY OUTPUT:")
    print(f"  {'─' * 55}")
    for line in advisory3.split("\n"):
        print(f"  {line}")

    print("\n\n" + "#" * 60)
    print("##  DEMO COMPLETE")
    print("##  All 3 scenarios demonstrate the advisory system logic.")
    print("##  When the server runs, /predict/advisory produces the same output.")
    print("#" * 60)
    print()


if __name__ == "__main__":
    run_demo()
