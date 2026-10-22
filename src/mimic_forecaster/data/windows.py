"""Subject-safe splits and window construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .mimic import HourlyStay
from .preprocessing import RobustStats, causal_impute, fit_robust_stats, transform


@dataclass
class WindowSet:
    values: np.ndarray
    observed_mask: np.ndarray
    feature_present_mask: np.ndarray
    age: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    drug_targets: np.ndarray
    drug_mask: np.ndarray
    drug_history: np.ndarray      # [context, drug] exposure the patient is already on (model input)
    start_targets: np.ndarray     # [drug] class newly started within start_horizon hours
    start_mask: np.ndarray        # [drug] scoreable: charted, not already active, future fully observed
    split: np.ndarray


def subject_split(subject_id: str, seed: int, train: float, val: float) -> str:
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    number = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if number < train else "val" if number < train + val else "test"


def fit_training_stats(stays: list[HourlyStay], seed: int, ratios: dict[str, float]) -> RobustStats:
    values, masks = [], []
    for stay in stays:
        if subject_split(stay.subject_id, seed, ratios["train"], ratios["val"]) == "train":
            values.append(np.asarray(stay.values, dtype=np.float32)); masks.append(np.asarray(stay.observed_mask, dtype=np.float32))
    if not values:
        raise ValueError("training split is empty")
    return fit_robust_stats(np.concatenate(values), np.concatenate(masks))


def drug_prevalence(windows: WindowSet, split: str = "train", start: bool = False) -> np.ndarray:
    """Per-class positive rate over labelled windows, for BCE positive weighting.

    Medication labels are heavily imbalanced (a vasopressor appears in a few percent of
    hours), so an unweighted BCE converges on predicting "never" for the rare classes.
    """
    keep = windows.split == split
    mask = (windows.start_mask if start else windows.drug_mask)[keep]
    total = mask.sum(0)
    positives = ((windows.start_targets if start else windows.drug_targets)[keep] * mask).sum(0)
    return np.divide(positives, total, out=np.zeros_like(positives), where=total > 0)


def build_windows(stays: list[HourlyStay], stats: RobustStats, *, context: int, horizon: int, stride: int,
                  max_gap, seed: int, ratios: dict[str, float], n_drugs: int = 0,
                  start_horizon: int = 24, forecast_features: np.ndarray | None = None) -> WindowSet:
    """`forecast_features` (0/1 per feature) limits which columns are forecast targets. Labs are
    inputs only: a next-hour lab value exists only when a draw happens to land in that hour,
    and those rare heavy-tailed errors were dragging the shared forecast head below
    repeat-last-value on every vital."""
    fields: dict[str, list[np.ndarray | str]] = {key: [] for key in WindowSet.__annotations__}
    for stay in stays:
        raw = np.asarray(stay.values, dtype=np.float32); mask = np.asarray(stay.observed_mask, dtype=np.float32)
        filled, age = causal_impute(raw, mask, stats.median, max_gap)
        scaled = transform(filled, stats)
        split = subject_split(stay.subject_id, seed, ratios["train"], ratios["val"])
        # A stay with no prescriptions join contributes no medication supervision, rather
        # than contributing all-negative labels the record does not support.
        has_drugs = n_drugs > 0 and len(stay.drug_active) == len(raw)
        active = np.asarray(stay.drug_active, dtype=np.float32) if has_drugs else None
        drug_present = (np.asarray(stay.drug_present_mask, dtype=np.float32) if has_drugs
                        else np.zeros(n_drugs, dtype=np.float32))
        last_start = len(raw) - context - horizon
        for start in range(0, last_start + 1, stride):
            target_slice = slice(start + context, start + context + horizon)
            target_mask = mask[target_slice] if forecast_features is None else mask[target_slice] * forecast_features
            if not target_mask.any():
                continue
            fields["values"].append(scaled[start:start + context])
            fields["observed_mask"].append(mask[start:start + context])
            fields["feature_present_mask"].append(np.asarray(stay.feature_present_mask, dtype=np.float32))
            fields["age"].append(age[start:start + context])
            fields["targets"].append(transform(np.where(target_mask, raw[target_slice], stats.median), stats))
            fields["target_mask"].append(target_mask)
            # Medication label: was the class active at any hour of the forecast horizon.
            fields["drug_targets"].append(active[target_slice].max(0) if active is not None
                                          else np.zeros(n_drugs, dtype=np.float32))
            fields["drug_mask"].append(drug_present)
            end = start + context
            if active is not None:
                fields["drug_history"].append(active[start:end])
                # New start: off at the last observed hour, on at some hour of the next
                # `start_horizon`. Windows whose future runs past the stay are unscored,
                # never scored negative.
                already = active[end - 1]
                complete = end + start_horizon <= len(raw)
                fields["start_targets"].append(((1.0 - already) * active[end:end + start_horizon].max(0))
                                               if complete else np.zeros(n_drugs, dtype=np.float32))
                fields["start_mask"].append(drug_present * (1.0 - already) * float(complete))
            else:
                fields["drug_history"].append(np.zeros((context, n_drugs), dtype=np.float32))
                fields["start_targets"].append(np.zeros(n_drugs, dtype=np.float32))
                fields["start_mask"].append(np.zeros(n_drugs, dtype=np.float32))
            fields["split"].append(split)
    if not fields["values"]:
        raise ValueError("no eligible windows were built")
    return WindowSet(**{
        key: np.asarray(value, dtype="U5" if key == "split" else np.float32)
        for key, value in fields.items()
    })
