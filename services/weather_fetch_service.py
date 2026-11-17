"""
Weather Fetch Service.

Solves the humidity sensor gap: the IoT hardware in this project only has
soil moisture, temperature, and NPK sensors -- no humidity sensor. Since
relative humidity is atmospheric (not soil-derived), it's reasonable to
source it from a weather API using the device's location instead of
requiring extra hardware.

Uses Open-Meteo (https://open-meteo.com) because it's free and requires no
API key, which keeps this from being a brittle external dependency. If the
request has no location, or the API call fails for any reason, we fall back
to a configured static value rather than letting the request fail -- a
prediction with a reasonable estimate is better than no prediction.

Results are cached briefly (cache_ttl_seconds in config) so a stream of
sensor readings from the same device doesn't hammer the weather API.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)

_cache: dict[tuple, tuple[float, float]] = {}  # (lat, lon) -> (timestamp, humidity)


def _round_coord(value: float) -> float:
    """Round to ~1km precision so nearby devices share a cache entry."""
    return round(value, 2)


def get_humidity(latitude: Optional[float], longitude: Optional[float]) -> tuple[float, str]:
    """
    Get a relative humidity percentage for the given coordinates.

    Returns:
        (humidity_percent, source) where source is one of:
        "weather_api", "cache", "fallback"
    """
    settings = ConfigLoader.load("app_settings.json")
    cfg = settings["sensor_fallbacks"]["humidity"]
    fallback = cfg["fallback_percent"]

    if latitude is None or longitude is None:
        logger.info("No location provided for humidity lookup; using fallback value.")
        return fallback, "fallback"

    key = (_round_coord(latitude), _round_coord(longitude))
    now = time.time()
    ttl = cfg.get("cache_ttl_seconds", 600)

    if key in _cache:
        cached_at, cached_value = _cache[key]
        if now - cached_at < ttl:
            return cached_value, "cache"

    try:
        response = requests.get(
            cfg["weather_api_base_url"],
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "relative_humidity_2m",
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        humidity = float(data["current"]["relative_humidity_2m"])
        _cache[key] = (now, humidity)
        return humidity, "weather_api"
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Weather API humidity fetch failed ({exc}); using fallback value.")
        return fallback, "fallback"


def get_ph_default() -> float:
    """Return the configured default pH used when no pH sensor reading is supplied."""
    settings = ConfigLoader.load("app_settings.json")
    return float(settings["sensor_fallbacks"]["ph"]["fallback_value"])
