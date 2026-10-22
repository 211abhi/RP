"""Convert MIMIC prescription orders into mask-aware hourly medication-exposure series.

One loader for every MIMIC release, same as `mimic.py`. Releases differ in column naming
(`starttime`/`stoptime` vs `startdate`/`enddate`) and time resolution (MIMIC-IV charts to
the second, MIMIC-III rounds prescriptions to the day). Those differences are absorbed
here by aliasing, so one registry and one set of weights serves all of them.

Prescriptions are keyed by `hadm_id`, not by ICU stay, so the join runs through the
`hadm_id` carried on each stay and is then clipped to that stay's hour grid.

Standard library only, so schema inspection works before torch is installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import yaml

from .mimic import _floor_hour, _open_csv, _parse_time, _resolve_column

START_ALIASES = ("starttime", "startdate")
STOP_ALIASES = ("stoptime", "enddate")


@dataclass(frozen=True)
class DrugSpec:
    name: str
    patterns: tuple[str, ...]


def load_drug_registry(path: str | Path) -> tuple[str, tuple[DrugSpec, ...], str]:
    """Return (version, specs, digest). Digest pins the checkpoint to this registry."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("drugs"), list):
        raise ValueError("drug registry must contain a drugs list")
    names: set[str] = set()
    patterns: set[str] = set()
    specs: list[DrugSpec] = []
    for entry in raw["drugs"]:
        name = str(entry["name"])
        matches = tuple(str(value).strip().lower() for value in entry["match"])
        if not matches or any(not value for value in matches):
            raise ValueError(f"drug class has an empty match pattern: {name}")
        if name in names:
            raise ValueError(f"duplicate drug class: {name}")
        overlap = patterns.intersection(matches)
        if overlap:
            raise ValueError(f"pattern claimed by two drug classes: {sorted(overlap)}")
        names.add(name)
        patterns.update(matches)
        specs.append(DrugSpec(name, matches))
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return str(raw.get("version", "unversioned")), tuple(specs), hashlib.sha256(payload).hexdigest()


def classify_drug(drug: str, specs: Sequence[DrugSpec]) -> int | None:
    """First matching class wins, so single-agent classes must precede grouped ones."""
    text = drug.strip().lower()
    if not text:
        return None
    for index, spec in enumerate(specs):
        if any(pattern in text for pattern in spec.patterns):
            return index
    return None


def collect_drug_intervals(
    path: str | Path,
    specs: Sequence[DrugSpec],
    allowed_hadm: set[str] | None = None,
    default_duration_hours: int = 24,
) -> tuple[dict[str, list[tuple[int, datetime, datetime]]], set[int]]:
    """Collect (drug_index, start, stop) exposure intervals per hadm_id.

    An order with no stop time is bounded to `default_duration_hours` rather than left
    open: an unbounded interval would mark a drug active for the rest of the admission and
    manufacture positives the record does not support.
    """
    intervals: dict[str, list[tuple[int, datetime, datetime]]] = defaultdict(list)
    seen: set[int] = set()
    with _open_csv(Path(path)) as handle:
        reader = csv.DictReader(handle)
        start_column = _resolve_column(reader.fieldnames, START_ALIASES, "prescription start")
        stop_column = _resolve_column(reader.fieldnames, STOP_ALIASES, "prescription stop")
        missing = {"hadm_id", "drug"}.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"prescriptions is missing required columns: {sorted(missing)}")
        has_type = "drug_type" in (reader.fieldnames or ())
        for row in reader:
            hadm_id = (row["hadm_id"] or "").strip()
            if not hadm_id or (allowed_hadm is not None and hadm_id not in allowed_hadm):
                continue
            # Additive/base rows describe the diluent of a MAIN order, not a second decision.
            if has_type and (row["drug_type"] or "MAIN").strip().upper() != "MAIN":
                continue
            index = classify_drug(row["drug"], specs)
            if index is None:
                continue
            try:
                start = _floor_hour(_parse_time(row[start_column]))
            except (TypeError, ValueError):
                continue
            try:
                stop = _floor_hour(_parse_time(row[stop_column]))
            except (TypeError, ValueError):
                stop = start + timedelta(hours=default_duration_hours)
            if stop < start:
                continue
            intervals[hadm_id].append((index, start, stop))
            seen.add(index)
    return dict(intervals), seen


def drug_hour_matrix(
    intervals: Sequence[tuple[int, datetime, datetime]],
    timestamps: Sequence[datetime],
    n_drugs: int,
) -> list[list[int]]:
    """Expand exposure intervals onto a stay's hour grid: 1 where the drug is active."""
    matrix = [[0] * n_drugs for _ in timestamps]
    if not timestamps or not intervals:
        return matrix
    first, last = timestamps[0], timestamps[-1]
    for index, start, stop in intervals:
        if stop < first or start > last:
            continue
        begin = max(0, int((start - first).total_seconds() // 3600))
        end = min(len(timestamps) - 1, int((stop - first).total_seconds() // 3600))
        for position in range(begin, end + 1):
            matrix[position][index] = 1
    return matrix
