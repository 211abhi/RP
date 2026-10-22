"""Contrastive alignment helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveAlignment(nn.Module):
    def __init__(self, ts_dim: int, text_dim: int, projection_dim: int, temperature: float = 0.07) -> None:
        super().__init__()
        self.ts_projection = nn.Linear(ts_dim, projection_dim)
        self.text_projection = nn.Linear(text_dim, projection_dim)
        self.temperature = temperature

    def forward(self, time_series: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if len(time_series) != len(text):
            raise ValueError("paired batches must have equal length")
        ts = F.normalize(self.ts_projection(time_series), dim=-1)
        tx = F.normalize(self.text_projection(text), dim=-1)
        logits = ts @ tx.T / self.temperature
        labels = torch.arange(len(ts), device=ts.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def trajectory_descriptions(values: torch.Tensor, observed: torch.Tensor, feature_names: list[str]) -> list[str]:
    """Generate identifier-free descriptions from pre-cutoff trajectories only."""
    descriptions: list[str] = []
    for sample, mask in zip(values.detach().cpu(), observed.detach().cpu()):
        parts = []
        for index, name in enumerate(feature_names):
            valid = sample[:, index][mask[:, index].bool()]
            if len(valid) < 2:
                state = "unavailable"
            else:
                delta = float(valid[-1] - valid[0])
                state = "rising" if delta > 0.5 else "falling" if delta < -0.5 else "stable"
            parts.append(f"{name} {state}")
        descriptions.append("ICU trajectory: " + ", ".join(parts) + ".")
    return descriptions

