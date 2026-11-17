"""
Weather Advisory Service.

Entirely rule-driven from config/weather_rules.json.
"""
from __future__ import annotations

from typing import List

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


def _evaluate_temperature(temperature: float, rules: List[dict]) -> List[str]:
    advices = []
    for rule in rules:
        condition = rule["condition"]
        if condition == "above" and temperature > rule["value_celsius"]:
            advices.append(rule["advice"].format(temperature=temperature))
        elif condition == "below" and temperature < rule["value_celsius"]:
            advices.append(rule["advice"].format(temperature=temperature))
        elif condition == "between" and rule["min_celsius"] <= temperature <= rule["max_celsius"]:
            advices.append(rule["advice"].format(temperature=temperature))
    return advices


def _evaluate_rainfall(rainfall_probability: float, rainfall_mm: float, rules: List[dict]) -> tuple[List[str], List[str]]:
    """Returns (advice_messages, recommended_actions). Rules are ordered high->low threshold."""
    sorted_rules = sorted(rules, key=lambda r: r["threshold_percent"], reverse=True)
    for rule in sorted_rules:
        if rainfall_probability >= rule["threshold_percent"]:
            advice = rule["advice"].format(
                rainfall_probability=rainfall_probability, rainfall_mm=rainfall_mm
            )
            return [advice], rule.get("actions", [])
    return [], []


def get_weather_advisory(temperature: float, rainfall_probability: float, rainfall_mm: float = 0.0) -> dict:
    """
    Generate weather-based farming advice from temperature and rainfall forecast.
    """
    rules = ConfigLoader.load("weather_rules.json")

    temp_advices = _evaluate_temperature(temperature, rules["temperature_rules"])
    rain_advices, actions = _evaluate_rainfall(rainfall_probability, rainfall_mm, rules["rainfall_rules"])

    all_advice = temp_advices + rain_advices

    return {
        "temperature": temperature,
        "rainfall_probability": rainfall_probability,
        "rainfall_mm": rainfall_mm,
        "advisories": all_advice,
        "recommended_actions": actions,
    }
