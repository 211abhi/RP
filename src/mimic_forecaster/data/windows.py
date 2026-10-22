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


def drug_prevalence(windows: WindowSet, split: str = "train") -> np.ndarray:
    """Per-class positive rate over labelled windows, for BCE positive weighting.

    Medication labels are heavily imbalanced (a vasopressor appears in a few percent of
    hours), so an unweighted BCE converges on predicting "never" for the rare classes.
    """
    keep = windows.split == split
    mask = windows.drug_mask[keep]
    total = mask.sum(0)
    positives = (windows.drug_targets[keep] * mask).sum(0)
    return np.divide(positives, total, out=np.zeros_like(positives), where=total > 0)


def build_windows(stays: list[HourlyStay], stats: RobustStats, *, context: int, horizon: int, stride: int,
                  max_gap: int, seed: int, ratios: dict[str, float], n_drugs: int = 0) -> WindowSet:
    fields: dict[str, list[np.ndarray | str]] = {key: [] for key in ("values", "observed_mask", "feature_present_mask", "age", "targets", "target_mask", "drug_targets", "drug_mask", "split")}
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
            target_mask = mask[target_slice]
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
            fields["split"].append(split)
    if not fields["values"]:
        raise ValueError("no eligible windows were built")
    return WindowSet(**{
        key: np.asarray(value, dtype="U5" if key == "split" else np.float32)
        for key, value in fields.items()
    })
