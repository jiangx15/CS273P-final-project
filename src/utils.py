"""Utility helpers for reproducible experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set random seed across common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a JSON serializable object to disk."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_device() -> torch.device:
    """Return an available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
