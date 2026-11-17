"""
API routes for the Smart Agriculture AI layer.

Endpoints:
    GET  /health
    POST /predict/crop
    POST /predict/fertilizer
    POST /predict/irrigation
    POST /predict/stage
    POST /predict/disease
    POST /predict/advisory
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from advisory.advisory_engine import generate_advisory
from api.schemas import (
    AdvisoryResponse,
    CropPredictionRequest,
    CropPredictionResponse,
    DiseasePredictionResponse,
    FertilizerRequest,
    FertilizerResponse,
    HealthResponse,
    IrrigationRequest,
    IrrigationResponse,
    PestRequest,
    PestResponse,
    StagePredictionResponse,
    WeatherAdvisoryRequest,
    WeatherAdvisoryResponse,
)
from inference.predict_crop import predict_crop
from inference.predict_disease import predict_disease
from inference.predict_stage import predict_stage
from services.fertilizer_service import recommend_fertilizer
from services.irrigation_service import recommend_irrigation
from services.pest_service import get_pest_advice
from services.weather_service import get_weather_advisory
from services.weather_fetch_service import get_humidity, get_ph_default
from utils.auth import verify_api_key
from utils.exceptions import AIBaseException
from utils.logger import get_logger
from utils.model_loader import ModelRegistry
from utils.model_versioning import get_active_version

logger = get_logger(__name__)

# /health is intentionally unauthenticated -- monitoring tools, Docker
# healthchecks, and load balancers typically need to probe it without
# credentials. Every other route requires the API key.
public_router = APIRouter()
router = APIRouter(dependencies=[Depends(verify_api_key)])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024  # 8 MB


async def _read_validated_image(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type '{image.content_type}'. Use JPEG or PNG.",
        )
    contents = await image.read()
    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image file too large. Maximum allowed size is 8 MB.",
        )
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image file.")
    return contents


@public_router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    loaded = []
    for key in [
        "crop_recommendation_model", "stage_model",
        "disease_model_rice", "disease_model_wheat",
    ]:
        try:
            ModelRegistry.get(key)
            loaded.append(key)
        except Exception:  # noqa: BLE001
            pass

    active_versions = {
        name: get_active_version(name)
        for name in ["crop_recommendation", "stage_model", "disease_model_rice", "disease_model_wheat"]
    }

    return HealthResponse(
        status="ok",
        models_loaded=ModelRegistry.is_loaded(),
        loaded_models=loaded,
        active_model_versions=active_versions,
    )


@router.post("/predict/crop", response_model=CropPredictionResponse, tags=["AI Modules"])
def predict_crop_endpoint(payload: CropPredictionRequest):
    try:
        if payload.humidity is not None:
            humidity, humidity_source = payload.humidity, "sensor"
        else:
            humidity, humidity_source = get_humidity(payload.latitude, payload.longitude)

        ph = payload.ph if payload.ph is not None else get_ph_default()

        result = predict_crop(
            n=payload.N, p=payload.P, k=payload.K,
            temperature=payload.temperature, humidity=humidity,
            moisture=payload.moisture, ph=ph,
        )
        result["humidity_used"] = humidity
        result["humidity_source"] = humidity_source
        result["ph_used"] = ph
        return CropPredictionResponse(**result)
    except AIBaseException as exc:
        logger.error(f"/predict/crop failed: {exc.message}")
        raise HTTPException(status_code=500, detail=exc.message) from exc


@router.post("/predict/fertilizer", response_model=FertilizerResponse, tags=["AI Modules"])
def predict_fertilizer_endpoint(payload: FertilizerRequest):
    try:
        result = recommend_fertilizer(payload.crop, payload.N, payload.P, payload.K, ph=payload.ph)
        return FertilizerResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"/predict/fertilizer failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/predict/irrigation", response_model=IrrigationResponse, tags=["AI Modules"])
def predict_irrigation_endpoint(payload: IrrigationRequest):
    try:
        result = recommend_irrigation(payload.crop, payload.stage, payload.soil_moisture)
        return IrrigationResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"/predict/irrigation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/predict/pest", response_model=PestResponse, tags=["AI Modules"])
def predict_pest_endpoint(payload: PestRequest):
    """
    Given a detected pest class name and current crop stage, returns complete
    management advice including chemical, organic, and cultural controls, with
    spray restrictions for the current growth stage automatically applied.
    """
    try:
        result = get_pest_advice(payload.pest_class, payload.crop, payload.stage)
        return PestResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"/predict/pest failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/predict/pest/list/{crop}", tags=["AI Modules"])
def list_pests_endpoint(crop: str):
    """List all known pests for a given crop (rice / wheat / maize)."""
    from services.pest_service import list_pests_for_crop
    return {"crop": crop, "pests": list_pests_for_crop(crop.strip().lower())}


@router.post("/predict/weather", response_model=WeatherAdvisoryResponse, tags=["AI Modules"])
def predict_weather_endpoint(payload: WeatherAdvisoryRequest):
    try:
        result = get_weather_advisory(
            payload.temperature, payload.rainfall_probability, payload.rainfall_mm
        )
        return WeatherAdvisoryResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"/predict/weather failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/predict/stage", response_model=StagePredictionResponse, tags=["AI Modules"])
async def predict_stage_endpoint(
    crop: str = Form(..., description="rice/wheat/maize"),
    image: UploadFile = File(...),
):
    try:
        contents = await _read_validated_image(image)
        result = predict_stage(contents, crop.strip().lower())
        return StagePredictionResponse(**result)
    except AIBaseException as exc:
        logger.error(f"/predict/stage failed: {exc.message}")
        raise HTTPException(status_code=500, detail=exc.message) from exc


@router.post("/predict/disease", response_model=DiseasePredictionResponse, tags=["AI Modules"])
async def predict_disease_endpoint(
    crop: str = Form(..., description="rice/wheat/maize"),
    image: UploadFile = File(...),
):
    try:
        contents = await _read_validated_image(image)
        result = predict_disease(contents, crop.strip().lower())
        return DiseasePredictionResponse(**result)
    except AIBaseException as exc:
        logger.error(f"/predict/disease failed: {exc.message}")
        raise HTTPException(status_code=500, detail=exc.message) from exc


@router.post("/predict/advisory", response_model=AdvisoryResponse, tags=["AI Modules"])
async def predict_advisory_endpoint(
    crop: str = Form(...),
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    soil_moisture: float = Form(...),
    temperature: float = Form(...),
    humidity: Optional[float] = Form(None),
    ph: Optional[float] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    rainfall_probability: float = Form(0.0),
    rainfall_mm: float = Form(0.0),
    known_stage: Optional[str] = Form(None),
    stage_image: Optional[UploadFile] = File(None),
    disease_image: Optional[UploadFile] = File(None),
):
    """
    The combined advisory endpoint. Sensor fields are required except
    `humidity` and `ph`, which are optional -- if your IoT device has no
    humidity or pH sensor, omit them. Humidity is then auto-fetched from a
    weather API using latitude/longitude (or a fallback default if no
    location is given); pH falls back to a neutral default. Both images are
    also optional; if omitted, that sub-module is skipped and `known_stage`
    (if supplied) is used for irrigation stage lookup.
    """
    try:
        stage_bytes = await _read_validated_image(stage_image) if stage_image else None
        disease_bytes = await _read_validated_image(disease_image) if disease_image else None

        resolved_humidity = humidity if humidity is not None else get_humidity(latitude, longitude)[0]

        result = generate_advisory(
            crop=crop.strip().lower(),
            n=N, p=P, k=K,
            soil_moisture=soil_moisture,
            temperature=temperature,
            humidity=resolved_humidity,
            ph=ph,
            rainfall_probability=rainfall_probability,
            rainfall_mm=rainfall_mm,
            stage_image_bytes=stage_bytes,
            disease_image_bytes=disease_bytes,
            known_stage=known_stage.strip().lower() if known_stage else None,
        )
        return AdvisoryResponse(**result)
    except AIBaseException as exc:
        logger.error(f"/predict/advisory failed: {exc.message}")
        raise HTTPException(status_code=500, detail=exc.message) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error(f"/predict/advisory failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc