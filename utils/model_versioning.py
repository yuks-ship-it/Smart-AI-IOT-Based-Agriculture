"""
Model versioning.

Every time a training script saves a model, it also calls
`register_version()` here, which:
  1. Copies the model file(s) into models/versions/<model_name>/<version_id>/
  2. Appends an entry to models/model_registry.json recording the version id,
     timestamp, algorithm, metrics, and file paths
  3. Marks that version as "active" (the one ModelRegistry/inference will load)

This gives you a rollback path: if a retrain produces a worse model, use
`list_versions()` to see history and `activate_version()` to point the live
model path back at a previous version, without retraining.

Note: the "live" paths declared in config/app_settings.json (e.g.
models/crop_recommendation.pkl) are always a COPY of whichever version is
currently active -- ModelRegistry only ever reads from those live paths, so
activating a version is just "copy that version's files over the live path."
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config_loader import resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

REGISTRY_FILE = "models/model_registry.json"
VERSIONS_DIR = "models/versions"


def _load_registry() -> Dict[str, Any]:
    path = resolve_path(REGISTRY_FILE)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: Dict[str, Any]) -> None:
    path = resolve_path(REGISTRY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_version(
    model_name: str,
    live_file_paths: List[str],
    metrics: Optional[Dict[str, Any]] = None,
    algorithm: Optional[str] = None,
) -> str:
    """
    Snapshot the just-trained model file(s) into a new version folder and
    mark this version as active.

    Args:
        model_name: logical name, e.g. "crop_recommendation", "stage_model"
        live_file_paths: the actual file paths just written (relative to
            project root) that should be snapshotted, e.g.
            ["models/crop_recommendation.pkl", "models/crop_scaler.pkl",
             "models/crop_label_encoder.pkl"]
        metrics: optional dict of evaluation metrics (accuracy, etc.)
        algorithm: optional algorithm name for display in version history

    Returns:
        the new version_id (timestamp-based string)
    """
    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = resolve_path(VERSIONS_DIR) / model_name / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    snapshotted_files = []
    for rel_path in live_file_paths:
        src = resolve_path(rel_path)
        if not src.exists():
            logger.warning(f"Expected model file not found, skipping snapshot: {src}")
            continue
        dst = version_dir / src.name
        shutil.copy2(src, dst)
        snapshotted_files.append(str(dst.relative_to(resolve_path("."))))

    registry = _load_registry()
    registry.setdefault(model_name, {"versions": [], "active_version": None})

    registry[model_name]["versions"].append({
        "version_id": version_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "metrics": metrics or {},
        "files": snapshotted_files,
    })
    registry[model_name]["active_version"] = version_id

    _save_registry(registry)
    logger.info(f"Registered model version '{version_id}' for '{model_name}' (now active).")
    return version_id


def list_versions(model_name: str) -> List[Dict[str, Any]]:
    registry = _load_registry()
    return registry.get(model_name, {}).get("versions", [])


def get_active_version(model_name: str) -> Optional[str]:
    registry = _load_registry()
    return registry.get(model_name, {}).get("active_version")


def activate_version(model_name: str, version_id: str, live_file_paths: List[str]) -> bool:
    """
    Roll back (or forward) to a specific previously-trained version by
    copying its snapshotted files back over the live model paths that
    ModelRegistry reads from. Call this, then restart the API process (or
    call ModelRegistry.load_all_models() again) to pick up the change.
    """
    registry = _load_registry()
    versions = registry.get(model_name, {}).get("versions", [])
    match = next((v for v in versions if v["version_id"] == version_id), None)

    if not match:
        logger.error(f"Version '{version_id}' not found for model '{model_name}'.")
        return False

    version_dir = resolve_path(VERSIONS_DIR) / model_name / version_id
    for rel_live_path in live_file_paths:
        live_path = resolve_path(rel_live_path)
        snapshot_path = version_dir / live_path.name
        if not snapshot_path.exists():
            logger.warning(f"Snapshot file missing for rollback: {snapshot_path}")
            continue
        live_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot_path, live_path)

    registry[model_name]["active_version"] = version_id
    _save_registry(registry)
    logger.info(f"Activated version '{version_id}' for '{model_name}'.")
    return True
