"""Leakage-safe imputation, scaling, and missingness channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustStats:
    median: np.ndarray
    scale: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {"median": self.median.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, value: dict[str, list[float]]) -> "RobustStats":
        return cls(np.asarray(value["median"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32))


def fit_robust_stats(values: np.ndarray, mask: np.ndarray) -> RobustStats:
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("values and mask must have matching [time, feature] shapes")
    medians, scales = [], []
    for index in range(values.shape[1]):
        observed = values[:, index][mask[:, index].astype(bool)]
        observed = observed[np.isfinite(observed)]
        if not observed.size:
            medians.append(0.0); scales.append(1.0); continue
        q25, median, q75 = np.percentile(observed, [25, 50, 75])
        medians.append(float(median)); scales.append(float(q75 - q25) or 1.0)
    return RobustStats(np.asarray(medians, dtype=np.float32), np.asarray(scales, dtype=np.float32))


def causal_impute(values: np.ndarray, mask: np.ndarray, fallback: np.ndarray, max_gap) -> tuple[np.ndarray, np.ndarray]:
    """Bounded forward fill followed by fallback; returns values and age channels.

    `max_gap` is one limit for every column or one per column (labs are held longer).
    """
    result = np.asarray(values, dtype=np.float32).copy()
    observed = np.asarray(mask, dtype=np.float32)
    if result.shape != observed.shape:
        raise ValueError("values and mask shapes differ")
    age = np.ones_like(result, dtype=np.float32)
    last = np.asarray(fallback, dtype=np.float32).copy()
    max_gap = np.broadcast_to(np.asarray(max_gap, dtype=np.int32), (result.shape[1],))
    since = max_gap + 1
    for time_index in range(result.shape[0]):
        for feature_index in range(result.shape[1]):
            if observed[time_index, feature_index] and np.isfinite(result[time_index, feature_index]):
                last[feature_index] = result[time_index, feature_index]
                since[feature_index] = 0
            else:
                since[feature_index] += 1
                result[time_index, feature_index] = last[feature_index] if since[feature_index] <= max_gap[feature_index] else fallback[feature_index]
            age[time_index, feature_index] = min(since[feature_index], max_gap[feature_index] + 1) / (max_gap[feature_index] + 1)
    return result, age


def transform(values: np.ndarray, stats: RobustStats) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) - stats.median) / stats.scale

