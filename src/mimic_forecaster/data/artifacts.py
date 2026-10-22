"""Prepared-array and metadata persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .preprocessing import RobustStats
from .windows import WindowSet


def save_prepared(directory: str | Path, windows: WindowSet, stats: RobustStats, metadata: dict) -> None:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "windows.npz", **asdict(windows))
    (directory / "preprocessing.json").write_text(json.dumps(stats.to_dict(), indent=2) + "\n")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_prepared(directory: str | Path) -> tuple[WindowSet, RobustStats, dict]:
    directory = Path(directory)
    with np.load(directory / "windows.npz", allow_pickle=False) as data:
        missing = [name for name in WindowSet.__annotations__ if name not in data]
        if missing:
            raise ValueError(f"prepared windows predate this task contract (missing {missing}); "
                             f"re-run mimic-prepare")
        windows = WindowSet(**{name: data[name] for name in WindowSet.__annotations__})
    stats = RobustStats.from_dict(json.loads((directory / "preprocessing.json").read_text()))
    metadata = json.loads((directory / "metadata.json").read_text())
    return windows, stats, metadata

