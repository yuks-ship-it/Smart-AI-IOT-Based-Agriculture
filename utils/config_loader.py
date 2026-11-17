"""
Centralized configuration loader.

All dynamic rules (fertilizer, irrigation, weather, disease treatments, app settings)
live as JSON files under config/. Nothing in this project hardcodes thresholds or
business rules in Python -- they are all loaded through this module so they can be
edited and reloaded without touching code.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"


class ConfigLoader:
    """Thread-safe, cached JSON config loader with manual reload support."""

    _cache: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def load(cls, filename: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load a JSON config file from the config/ directory.

        Args:
            filename: e.g. "fertilizer_rules.json"
            force_reload: bypass cache and re-read from disk

        Returns:
            Parsed JSON content as a dict.
        """
        if not force_reload and filename in cls._cache:
            return cls._cache[filename]

        path = CONFIG_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with cls._lock:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cls._cache[filename] = data
        return data

    @classmethod
    def reload_all(cls) -> None:
        """Force-reload every previously loaded config file from disk."""
        filenames = list(cls._cache.keys())
        for filename in filenames:
            cls.load(filename, force_reload=True)

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()


def get_base_dir() -> Path:
    return BASE_DIR


def get_config_dir() -> Path:
    return CONFIG_DIR


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root (BASE_DIR)."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return BASE_DIR / relative_path
