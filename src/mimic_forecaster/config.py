"""Configuration loading, validation, and reproducible digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("configuration root must be a mapping")
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent.parent)
    validate_config(cfg)
    return cfg


def resolve_path(cfg: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(cfg["_project_root"]) / path


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ("project", "data", "task", "tokenizer", "encoder", "training"):
        if section not in cfg:
            raise ValueError(f"missing configuration section: {section}")
    ratios = cfg["data"]["split_ratios"]
    if set(ratios) != {"train", "val", "test"}:
        raise ValueError("split_ratios must define train, val, and test")
    if abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    context = int(cfg["task"]["context_hours"])
    patch = int(cfg["tokenizer"]["patch_size"])
    if context < 1 or patch < 1 or context % patch:
        raise ValueError("context_hours must be positive and divisible by patch_size")
    if int(cfg["task"]["horizon_hours"]) < 1:
        raise ValueError("horizon_hours must be positive")
    rate = float(cfg["task"].get("column_dropout_rate", 0.0))
    if not 0.0 <= rate < 1.0:
        raise ValueError("column_dropout_rate must be in [0, 1)")
    weights = cfg["task"].get("loss_weights", {}) or {}
    unknown = set(weights).difference({"forecast", "drug", "impute"})
    if unknown:
        raise ValueError(f"unknown loss_weights keys: {sorted(unknown)}")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("loss weights must be non-negative")
    if weights and not any(float(value) > 0 for value in weights.values()):
        raise ValueError("at least one loss weight must be positive")
    if float(weights.get("impute", 0.0)) > 0 and rate <= 0:
        raise ValueError("the impute head needs task.column_dropout_rate > 0 to be supervised")


def config_digest(cfg: dict[str, Any]) -> str:
    clean = {key: value for key, value in cfg.items() if not key.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

