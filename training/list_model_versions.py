"""
View and roll back trained model versions.

Usage:
    # List all versions of a model, newest first
    python training/list_model_versions.py crop_recommendation
    python training/list_model_versions.py stage_model
    python training/list_model_versions.py disease_model

    # Roll back to a specific previous version
    python training/list_model_versions.py crop_recommendation --activate 20260615T120000Z

After activating a different version, restart the API process (or call
ModelRegistry.load_all_models() again) for the change to take effect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config_loader import ConfigLoader
from utils.model_versioning import activate_version, get_active_version, list_versions

LIVE_PATHS_BY_MODEL = {
    "crop_recommendation": ["path", "scaler_path", "label_encoder_path"],
    "stage_model": ["path", "labels_path"],
    "disease_model": ["path", "labels_path"],
}

CONFIG_KEY_BY_MODEL = {
    "crop_recommendation": "crop_recommendation",
    "stage_model": "stage_classification",
    "disease_model": "disease_detection",
}


def _live_paths_for(model_name: str) -> list[str]:
    settings = ConfigLoader.load("app_settings.json")
    cfg = settings["models"][CONFIG_KEY_BY_MODEL[model_name]]
    return [cfg[key] for key in LIVE_PATHS_BY_MODEL[model_name]]


def main():
    parser = argparse.ArgumentParser(description="List or roll back model versions.")
    parser.add_argument("model_name", choices=list(LIVE_PATHS_BY_MODEL.keys()))
    parser.add_argument("--activate", type=str, default=None, help="Version ID to roll back to.")
    args = parser.parse_args()

    if args.activate:
        live_paths = _live_paths_for(args.model_name)
        success = activate_version(args.model_name, args.activate, live_paths)
        if success:
            print(f"Activated version {args.activate} for {args.model_name}. Restart the API to apply.")
        else:
            print(f"Failed to activate version {args.activate}. Check logs above.")
        return

    versions = list_versions(args.model_name)
    active = get_active_version(args.model_name)

    if not versions:
        print(f"No versions recorded yet for '{args.model_name}'. Train it first.")
        return

    print(f"\nVersion history for '{args.model_name}' (active: {active})\n")
    for v in reversed(versions):  # newest first
        marker = " <- ACTIVE" if v["version_id"] == active else ""
        print(f"  {v['version_id']}  algo={v.get('algorithm')}  metrics={v.get('metrics')}{marker}")
    print(f"\nTo roll back: python training/list_model_versions.py {args.model_name} --activate <version_id>\n")


if __name__ == "__main__":
    main()
