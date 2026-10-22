"""Versioned canonical feature registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from .mimic import FeatureSpec


def load_registry(path: str | Path) -> tuple[str, tuple[FeatureSpec, ...], str]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("features"), list):
        raise ValueError("feature registry must contain a features list")
    names: set[str] = set()
    itemids: set[int] = set()
    features: list[FeatureSpec] = []
    for entry in raw["features"]:
        name = str(entry["name"])
        ids = tuple(int(value) for value in entry["itemids"])
        low, high = (float(value) for value in entry["valid_range"])
        if name in names or any(value in itemids for value in ids):
            raise ValueError(f"duplicate feature name or itemid: {name}")
        if not ids or low >= high:
            raise ValueError(f"invalid feature specification: {name}")
        names.add(name)
        itemids.update(ids)
        table = str(entry.get("table", "chartevents"))
        if table not in ("chartevents", "labevents"):
            raise ValueError(f"unknown table for {name}: {table}")
        fill = entry.get("max_fill_hours")
        features.append(FeatureSpec(name, ids, low, high, table, int(fill) if fill is not None else None))
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return str(raw.get("version", "unversioned")), tuple(features), hashlib.sha256(payload).hexdigest()



def fill_limits(features, default: int):
    """Per-feature forward-fill limit: labs are drawn daily, vitals hourly."""
    import numpy as np
    return np.asarray([f.max_fill_hours if f.max_fill_hours is not None else default for f in features], dtype=np.int32)
