"""Pydantic request/response schemas for all API endpoints."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Crop Recommendation
# ---------------------------------------------------------------------------

class CropPredictionRequest(BaseModel):
    N: float = Field(..., ge=0, le=300, description="Nitrogen level")
    P: float = Field(..., ge=0, le=300, description="Phosphorus level")
    K: float = Field(..., ge=0, le=300, description="Potassium level")
    temperature: float = Field(..., ge=-10, le=60, description="Temperature in Celsius")
    moisture: float = Field(..., ge=0, le=100, description="Soil moisture %")
    humidity: Optional[float] = Field(
        None, ge=0, le=100,
        description="Relative humidity %. Optional -- if your IoT device has no "
                     "humidity sensor, omit this and supply latitude/longitude "
                     "instead; humidity will be auto-fetched from a weather API.",
    )
    ph: Optional[float] = Field(
        None, ge=0, le=14,
        description="Soil pH. Optional -- if your IoT device has no pH sensor, "
                     "omit this and a neutral default will be used.",
    )
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Device latitude, used to auto-fetch humidity if not supplied")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Device longitude, used to auto-fetch humidity if not supplied")
    device_id: Optional[str] = None
    timestamp: Optional[str] = None


class CropTopResult(BaseModel):
    crop: str
    probability: float


class CropPredictionResponse(BaseModel):
    recommended_crop: str
    confidence: float
    top_3: List[CropTopResult]
    humidity_used: float
    humidity_source: str
    ph_used: float


# ---------------------------------------------------------------------------
# Fertilizer Recommendation
# ---------------------------------------------------------------------------

class FertilizerRequest(BaseModel):
    crop: str = Field(..., description="Crop type, e.g. rice/wheat/maize")
    N: float = Field(..., ge=0, le=300)
    P: float = Field(..., ge=0, le=300)
    K: float = Field(..., ge=0, le=300)
    ph: Optional[float] = Field(None, ge=0, le=14, description="Soil pH, optional. Adds lime/sulfur guidance if provided.")

    @field_validator("crop")
    @classmethod
    def normalize_crop(cls, v: str) -> str:
        return v.strip().lower()


class FertilizerResponse(BaseModel):
    crop: str
    npk_input: dict
    optimal_npk: dict
    recommendations: List[str]
    details: List[dict]


# ---------------------------------------------------------------------------
# Irrigation Recommendation
# ---------------------------------------------------------------------------

class IrrigationRequest(BaseModel):
    crop: str
    stage: str = Field(..., description="seedling/tillering/vegetative/flowering/maturity")
    soil_moisture: float = Field(..., ge=0, le=100)

    @field_validator("crop", "stage")
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.strip().lower()


class IrrigationResponse(BaseModel):
    crop: str
    stage: str
    soil_moisture: float
    min_moisture_required: float
    ideal_moisture: float
    irrigation_required: bool
    recommended_amount_mm: float
    message: str


# ---------------------------------------------------------------------------
# Growth Stage (image upload handled via multipart/form-data in the route,
# this schema documents the non-file fields and the response shape)
# ---------------------------------------------------------------------------

class StagePredictionResponse(BaseModel):
    crop: str
    growth_stage: str
    confidence: float
    all_probabilities: dict
    low_confidence_warning: bool


# ---------------------------------------------------------------------------
# Disease Detection
# ---------------------------------------------------------------------------

class DiseaseTreatmentDetail(BaseModel):
    chemical: str = "N/A"
    organic: str = "N/A"
    systemic: str = "N/A"


class DiseasePredictionResponse(BaseModel):
    crop: str
    disease: str
    display_name: str
    also_known_as: Optional[str] = ""
    confidence: float
    severity: str
    treatment: DiseaseTreatmentDetail
    treatment_summary: str
    prevention: str
    symptoms: str = ""
    causal_agent: str = "Unknown"
    favorable_conditions: str = "Unknown"
    low_confidence_warning: bool


# ---------------------------------------------------------------------------
# Weather Advisory (used internally by /predict/advisory; exposed standalone too)
# ---------------------------------------------------------------------------

class WeatherAdvisoryRequest(BaseModel):
    temperature: float = Field(..., ge=-10, le=60)
    rainfall_probability: float = Field(0.0, ge=0, le=100)
    rainfall_mm: float = Field(0.0, ge=0)


class WeatherAdvisoryResponse(BaseModel):
    temperature: float
    rainfall_probability: float
    rainfall_mm: float
    advisories: List[str]
    recommended_actions: List[str]


# ---------------------------------------------------------------------------
# Combined Advisory (the non-file fields; images are optional multipart uploads)
# ---------------------------------------------------------------------------

class AdvisoryResponse(BaseModel):
    crop: str
    crop_stage: Optional[dict] = None
    disease: Optional[dict] = None
    fertilizer: dict
    irrigation: dict
    weather_advice: dict
    conflicts: Optional[List[dict]] = None
    final_advisory: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    loaded_models: List[str]
    active_model_versions: Optional[dict] = None


# ---------------------------------------------------------------------------
# Pest Recommendation
# ---------------------------------------------------------------------------

class PestRequest(BaseModel):
    pest_class: str = Field(
        ...,
        description="Pest class name matching training folder, e.g. rice_yellow_stem_borer"
    )
    crop: str = Field(..., description="rice / wheat / maize")
    stage: str = Field(
        ...,
        description="seedling / tillering / vegetative / flowering / maturity"
    )

    @field_validator("crop", "stage", "pest_class")
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.strip().lower()


class PestResponse(BaseModel):
    pest_class: str
    display_name: str
    scientific_name: Optional[str]
    crop: str
    stage: str
    severity: str
    economic_threshold: Optional[str]
    chemical_control: Optional[str]
    organic_control: Optional[str]
    cultural_control: Optional[str]
    stage_specific_advice: Optional[str]
    spray_restrictions: List[str]
    prevention: Optional[str]
    summary: str