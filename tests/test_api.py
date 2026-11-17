"""
Tests for rule-based services and API endpoints.

Run with:
    pytest tests/ -v

Note: tests for /predict/stage, /predict/disease, and /predict/advisory with
images require trained .h5 models to be present (see training/ scripts).
Those tests are skipped automatically if the models aren't found, so this
suite is safe to run immediately after cloning, before any training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import app
from services.fertilizer_service import recommend_fertilizer
from services.irrigation_service import recommend_irrigation
from services.weather_service import get_weather_advisory
from utils.config_loader import resolve_path

client = TestClient(app)


# ---------------------------------------------------------------------------
# Service-level unit tests (no models required)
# ---------------------------------------------------------------------------

def test_fertilizer_service_detects_deficiency():
    result = recommend_fertilizer("rice", n=50, p=20, k=20)
    assert result["crop"] == "rice"
    assert len(result["recommendations"]) > 0
    assert any(d["status"] == "deficient" for d in result["details"])


def test_fertilizer_service_no_deficiency():
    result = recommend_fertilizer("rice", n=100, p=50, k=50)
    assert "No additional fertilizer required" in result["recommendations"][0]


def test_irrigation_service_required():
    result = recommend_irrigation("rice", "flowering", soil_moisture=20)
    assert result["irrigation_required"] is True
    assert result["recommended_amount_mm"] > 0


def test_irrigation_service_not_required():
    result = recommend_irrigation("rice", "maturity", soil_moisture=80)
    assert result["irrigation_required"] is False


def test_weather_service_high_rain_temp():
    result = get_weather_advisory(temperature=40, rainfall_probability=80, rainfall_mm=25)
    assert len(result["advisories"]) >= 1
    assert "delay_fertilizer" in result["recommended_actions"]


# ---------------------------------------------------------------------------
# API-level tests
# ---------------------------------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_fertilizer_endpoint():
    response = client.post(
        "/predict/fertilizer", json={"crop": "wheat", "N": 60, "P": 30, "K": 20}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["crop"] == "wheat"
    assert "recommendations" in body


def test_irrigation_endpoint():
    response = client.post(
        "/predict/irrigation",
        json={"crop": "maize", "stage": "vegetative", "soil_moisture": 25},
    )
    assert response.status_code == 200
    assert response.json()["irrigation_required"] is True


def test_weather_endpoint():
    response = client.post(
        "/predict/weather",
        json={"temperature": 15, "rainfall_probability": 10, "rainfall_mm": 0},
    )
    assert response.status_code == 200


@pytest.mark.skipif(
    not resolve_path("models/crop_recommendation.pkl").exists(),
    reason="Crop recommendation model not trained yet. Run training/train_crop_model.py first.",
)
def test_crop_prediction_endpoint():
    response = client.post(
        "/predict/crop",
        json={
            "N": 90, "P": 42, "K": 43,
            "temperature": 25, "humidity": 80, "moisture": 60,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "recommended_crop" in body
    assert 0 <= body["confidence"] <= 1
