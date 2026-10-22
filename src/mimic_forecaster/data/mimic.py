"""Convert MIMIC ICU chart events into fixed-width, mask-aware hourly series.

One loader for every MIMIC release. Releases differ in column naming (`stay_id` vs
`icustay_id`), file conventions (gzipped lowercase vs plain uppercase), and quality flags
(MIMIC-III carries an `error` column that MIMIC-IV dropped). Those differences are
absorbed here, by aliasing, so a single registry, tensor shape, and set of weights serves
all of them. Which itemid means which vital is registry data, never code.

This module intentionally uses only the Python standard library so schema inspection and
the first data-contract smoke test work before the ML environment is installed.
"""

from __future__ import annotations

import csv
import gzip
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping, Sequence, TextIO

if TYPE_CHECKING:
    from .prescriptions import DrugSpec


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    itemids: tuple[int, ...]
    minimum: float
    maximum: float
    table: str = "chartevents"            # or "labevents"
    max_fill_hours: int | None = None     # forward-fill limit; None -> data.max_forward_fill_hours


DEFAULT_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("heart_rate", (220045,), 20.0, 260.0),
    FeatureSpec("respiratory_rate", (220210,), 2.0, 80.0),
    FeatureSpec("spo2", (220277,), 40.0, 100.0),
    FeatureSpec("systolic_bp", (220179,), 30.0, 300.0),
    FeatureSpec("diastolic_bp", (220180,), 10.0, 200.0),
    FeatureSpec("mean_bp", (220181,), 20.0, 250.0),
    FeatureSpec("temperature_f", (223761,), 80.0, 115.0),
)


@dataclass
class HourlyStay:
    """Canonical data for one ICU stay.

    `values` remains NaN where a cell is unobserved. Imputation and scaling are separate,
    train-fitted steps. `observed_mask` is never changed by later imputation.

    The medication fields follow the same three-way missingness contract as the vitals:
    `drug_active` is the per-hour exposure matrix, and `drug_present_mask` records whether
    the source charted that drug class at all, so a release with no prescriptions file is
    masked out of the medication loss instead of training it towards all-zero labels.
    """

    subject_id: str
    stay_id: str
    timestamps: list[datetime]
    feature_names: tuple[str, ...]
    values: list[list[float]]
    observed_mask: list[list[int]]
    feature_present_mask: list[int]
    hadm_id: str = ""
    drug_names: tuple[str, ...] = ()
    drug_active: list[list[int]] = field(default_factory=list)
    drug_present_mask: list[int] = field(default_factory=list)


STAY_ID_ALIASES = ("stay_id", "icustay_id")


def _resolve_column(fieldnames: Iterable[str] | None, aliases: Sequence[str], purpose: str) -> str:
    """Pick whichever spelling of a column this MIMIC release uses."""
    available = set(fieldnames or ())
    for alias in aliases:
        if alias in available:
            return alias
    raise ValueError(f"no {purpose} column found: tried {list(aliases)}, have {sorted(available)}")


def _open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def _floor_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def _hours(start: datetime, end: datetime) -> Iterator[datetime]:
    current = _floor_hour(start)
    last = _floor_hour(end)
    while current <= last:
        yield current
        current += timedelta(hours=1)


def load_stay_bounds(path: str | Path, source: str = "") -> dict[str, tuple[str, datetime, datetime, str]]:
    """Return stay_id -> (subject_id, intime, outtime, hadm_id).

    `source` namespaces the subject id so that subjects from different releases can never
    collide and leak the same person across a train/test boundary.

    `hadm_id` is the join key for prescriptions, which are ordered per admission rather
    than per ICU stay. It is optional: a source without the column yields "" and simply
    carries no medication labels.
    """
    result: dict[str, tuple[str, datetime, datetime, str]] = {}
    with _open_csv(Path(path)) as handle:
        reader = csv.DictReader(handle)
        stay_column = _resolve_column(reader.fieldnames, STAY_ID_ALIASES, "stay id")
        missing = {"subject_id", "intime", "outtime"}.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"icustays is missing required columns: {sorted(missing)}")
        has_hadm = "hadm_id" in (reader.fieldnames or ())
        for row in reader:
            if not row[stay_column] or not row["intime"] or not row["outtime"]:
                continue
            subject = f"{source}:{row['subject_id']}" if source else row["subject_id"]
            result[row[stay_column]] = (
                subject, _parse_time(row["intime"]), _parse_time(row["outtime"]),
                (row["hadm_id"] or "").strip() if has_hadm else "",
            )
    return result


def collect_hourly_events(
    path: str | Path,
    features: Sequence[FeatureSpec] = DEFAULT_FEATURES,
    allowed_stays: set[str] | None = None,
) -> tuple[dict[tuple[str, datetime, int], list[float]], set[int]]:
    """Collect valid numeric observations by stay, hour, and canonical feature index."""
    item_map = {itemid: (index, spec) for index, spec in enumerate(features) for itemid in spec.itemids
                if spec.table == "chartevents"}
    buckets: dict[tuple[str, datetime, int], list[float]] = defaultdict(list)
    seen_items: set[int] = set()
    with _open_csv(Path(path)) as handle:
        reader = csv.DictReader(handle)
        stay_column = _resolve_column(reader.fieldnames, STAY_ID_ALIASES, "stay id")
        missing = {"charttime", "itemid", "valuenum"}.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"chartevents is missing required columns: {sorted(missing)}")
        has_error_flag = "error" in (reader.fieldnames or ())  # MIMIC-III only
        for row in reader:
            stay_id = row[stay_column]
            if allowed_stays is not None and stay_id not in allowed_stays:
                continue
            if has_error_flag and row["error"] == "1":
                continue
            try:
                itemid = int(row["itemid"])
                index, spec = item_map[itemid]
                value = float(row["valuenum"])
                charttime = _floor_hour(_parse_time(row["charttime"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value) or not (spec.minimum <= value <= spec.maximum):
                continue
            buckets[(stay_id, charttime, index)].append(value)
            seen_items.add(itemid)
    return buckets, seen_items


def collect_lab_events(
    path: str | Path,
    features: Sequence[FeatureSpec],
    windows: Mapping[str, Sequence[tuple[str, datetime, datetime]]],
) -> tuple[dict[tuple[str, datetime, int], list[float]], set[int]]:
    """Collect lab results onto ICU stays by subject and time.

    labevents carries no stay id, and `hadm_id` is blank for many MIMIC-III rows, so a
    result belongs to whichever of the subject's stays was open at that hour. Results drawn
    outside every stay (clinic, ED before transfer) have no hour to land on and are dropped.
    """
    item_map = {itemid: (index, spec) for index, spec in enumerate(features) for itemid in spec.itemids
                if spec.table == "labevents"}
    buckets: dict[tuple[str, datetime, int], list[float]] = defaultdict(list)
    seen_items: set[int] = set()
    if not item_map:
        return buckets, seen_items
    with _open_csv(Path(path)) as handle:
        reader = csv.DictReader(handle)
        missing = {"subject_id", "charttime", "itemid", "valuenum"}.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"labevents is missing required columns: {sorted(missing)}")
        for row in reader:
            stays = windows.get(row["subject_id"])
            if not stays:
                continue
            try:
                index, spec = item_map[int(row["itemid"])]
                value = float(row["valuenum"])
                charttime = _floor_hour(_parse_time(row["charttime"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value) or not (spec.minimum <= value <= spec.maximum):
                continue
            for stay_id, start, end in stays:
                if start <= charttime <= end:
                    buckets[(stay_id, charttime, index)].append(value)
                    seen_items.add(int(row["itemid"]))
                    break
    return buckets, seen_items


def prepare_stays(
    icustays_path: str | Path,
    chartevents_path: str | Path,
    features: Sequence[FeatureSpec] = DEFAULT_FEATURES,
    limit_stays: int | None = None,
    source: str = "",
    prescriptions_path: str | Path | None = None,
    drugs: Sequence["DrugSpec"] = (),
    default_drug_hours: int = 24,
    labevents_path: str | Path | None = None,
) -> list[HourlyStay]:
    """Create deterministic hourly series for ICU stays using within-hour medians.

    When `prescriptions_path` and `drugs` are supplied, each stay also carries the hourly
    medication-exposure matrix for the canonical drug classes. Omitting them leaves the
    medication fields empty, which masks the stay out of the medication loss.
    """
    from .prescriptions import collect_drug_intervals, drug_hour_matrix

    bounds = load_stay_bounds(icustays_path, source)
    selected_ids = sorted(bounds, key=lambda value: int(value))
    if limit_stays is not None:
        if limit_stays < 1:
            raise ValueError("limit_stays must be positive")
        selected_ids = selected_ids[:limit_stays]
    selected = set(selected_ids)
    buckets, seen_items = collect_hourly_events(chartevents_path, features, selected)
    seen = {"chartevents": seen_items, "labevents": set()}
    if labevents_path is not None:
        windows: dict[str, list[tuple[str, datetime, datetime]]] = defaultdict(list)
        for stay_id in selected_ids:
            subject, intime, outtime, _ = bounds[stay_id]
            raw_subject = subject.split(":", 1)[1] if source else subject
            windows[raw_subject].append((stay_id, _floor_hour(intime), _floor_hour(outtime)))
        labs, seen["labevents"] = collect_lab_events(labevents_path, features, windows)
        for key, observations in labs.items():
            buckets.setdefault(key, []).extend(observations)
    present = [int(any(itemid in seen[spec.table] for itemid in spec.itemids)) for spec in features]

    intervals: dict[str, list] = {}
    drug_present = [0] * len(drugs)
    if prescriptions_path is not None and drugs:
        allowed_hadm = {bounds[stay_id][3] for stay_id in selected_ids if bounds[stay_id][3]}
        intervals, seen_drugs = collect_drug_intervals(
            prescriptions_path, drugs, allowed_hadm or None, default_drug_hours)
        drug_present = [int(index in seen_drugs) for index in range(len(drugs))]
    drug_names = tuple(spec.name for spec in drugs)

    output: list[HourlyStay] = []
    for stay_id in selected_ids:
        subject_id, intime, outtime, hadm_id = bounds[stay_id]
        timestamps = list(_hours(intime, outtime))
        values: list[list[float]] = []
        masks: list[list[int]] = []
        for timestamp in timestamps:
            row = [math.nan] * len(features)
            mask = [0] * len(features)
            for index in range(len(features)):
                observations = buckets.get((stay_id, timestamp, index))
                if observations:
                    row[index] = float(median(observations))
                    mask[index] = 1
            values.append(row)
            masks.append(mask)
        active = (drug_hour_matrix(intervals.get(hadm_id, ()), timestamps, len(drugs))
                  if drug_names else [])
        output.append(HourlyStay(
            subject_id=subject_id,
            stay_id=stay_id,
            timestamps=timestamps,
            feature_names=tuple(spec.name for spec in features),
            values=values,
            observed_mask=masks,
            feature_present_mask=present.copy(),
            hadm_id=hadm_id,
            drug_names=drug_names,
            drug_active=active,
            drug_present_mask=drug_present.copy() if drug_names else [],
        ))
    return output



def load_sources(cfg: dict, features: Sequence[FeatureSpec],
                 limit_stays: int | None = None,
                 drugs: Sequence["DrugSpec"] = ()) -> list[HourlyStay]:
    """Load every configured MIMIC release into one pooled, source-namespaced cohort.

    `feature_present_mask` and `drug_present_mask` are fitted per source, so a release that
    never charts a vital, or has no prescriptions file, is marked absent for its stays
    instead of silently looking like a run of missing values or of negative drug labels.
    """
    from ..config import resolve_path

    stays: list[HourlyStay] = []
    for source in config_sources(cfg):
        root = resolve_path(cfg, source["root"])
        prescriptions = root / source["prescriptions"] if source.get("prescriptions") else None
        labevents = root / source["labevents"] if source.get("labevents") else None
        for table, path in (("prescriptions", prescriptions), ("labevents", labevents)):
            if path is not None and not path.exists():
                raise ValueError(f"configured {table} file is missing: {path}")
        stays.extend(prepare_stays(root / source["icustays"], root / source["chartevents"],
                                   features, limit_stays, source["name"],
                                   prescriptions, drugs,
                                   int(cfg["data"].get("default_drug_hours", 24)), labevents))
    if not stays:
        raise ValueError("no stays loaded from any configured source")
    return stays


def config_sources(cfg: dict) -> list[dict]:
    """Read the `data.sources` list, accepting the single-source config shape too."""
    data = cfg["data"]
    if "sources" in data:
        sources = data["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("data.sources must be a non-empty list")
        names = [str(entry["name"]) for entry in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"data.sources names must be unique: {names}")
        return [dict(entry, name=str(entry["name"])) for entry in sources]
    return [{"name": "", "root": data["root"], "icustays": data["icustays"],
             "chartevents": data["chartevents"],
             **{key: data[key] for key in ("prescriptions", "labevents") if key in data}}]
