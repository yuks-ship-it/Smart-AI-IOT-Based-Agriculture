"""
Smart Agriculture AI Layer - FastAPI entry point.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload   (dev)
    uvicorn main:app --host 0.0.0.0 --port 8000             (prod, behind a process manager)

All models are loaded ONCE at startup via the lifespan handler below, then
served from the in-memory ModelRegistry for every request.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import public_router, router
from utils.config_validator import validate_all_configs
from utils.exceptions import AIBaseException
from utils.logger import get_logger
from utils.model_loader import ModelRegistry
from utils.rate_limiter import RateLimitMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Smart Agriculture AI Layer...")
    try:
        validate_all_configs()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Config validation error at startup: {exc}")
        raise  # abort startup if configs are broken
    try:
        ModelRegistry.load_all_models()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error during model loading: {exc}")
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down Smart Agriculture AI Layer.")


app = FastAPI(
    title="Smart Agriculture AI Layer",
    description=(
        "AI microservice providing crop recommendation, growth stage "
        "detection, disease detection, fertilizer/irrigation recommendation, "
        "weather advisory, and a combined advisory engine for a Smart "
        "Agriculture IoT system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: origins are read from ALLOWED_ORIGINS in .env (comma-separated).
# Defaults to localhost for safe local development. Set this to your actual
# dashboard domain(s) before deploying -- do NOT leave it as "*" in production.
_allowed_origins_raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost,http://localhost:3000,http://127.0.0.1:3000")
_allowed_origins = [origin.strip() for origin in _allowed_origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)


@app.exception_handler(AIBaseException)
async def ai_exception_handler(request: Request, exc: AIBaseException):
    logger.error(f"Unhandled AIBaseException on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details},
    )


app.include_router(public_router)
app.include_router(router)

# Mount static assets (CSS, JS, images) for the dashboard
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", tags=["Dashboard"], include_in_schema=False)
@app.get("/dashboard", tags=["Dashboard"], include_in_schema=False)
def serve_dashboard():
    """Serve the AgriMind AI Companion dashboard."""
    dashboard_html = _static_dir / "index.html"
    if dashboard_html.exists():
        return FileResponse(str(dashboard_html), media_type="text/html")
    return {
        "service": "Smart Agriculture AI Layer",
        "status": "running",
        "docs": "/docs",
        "dashboard": "Dashboard HTML not found. Ensure static/index.html exists.",
    }
