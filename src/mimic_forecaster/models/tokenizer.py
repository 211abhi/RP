"""Mask-aware patch tokenizer."""

from __future__ import annotations

import torch
from torch import nn


class MaskAwarePatchTokenizer(nn.Module):
    def __init__(self, n_features: int, patch_size: int, embed_dim: int, *, include_age: bool = True) -> None:
        super().__init__()
        self.n_features = n_features
        self.patch_size = patch_size
        self.include_age = include_age
        channels = 3 + int(include_age)  # values, observed, feature-present, optional age
        self.input_dim = patch_size * n_features * channels
        self.projection = nn.Sequential(nn.Linear(self.input_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, values: torch.Tensor, observed: torch.Tensor, present: torch.Tensor,
                age: torch.Tensor | None = None) -> torch.Tensor:
        if values.ndim != 3 or values.shape != observed.shape:
            raise ValueError("values and observed must match [batch, time, feature]")
        batch, time, features = values.shape
        if features != self.n_features or time % self.patch_size:
            raise ValueError("input shape is incompatible with tokenizer contract")
        if present.shape != (batch, features):
            raise ValueError("present mask must be [batch, feature]")
        channels = [values, observed, present[:, None, :].expand(-1, time, -1)]
        if self.include_age:
            if age is None or age.shape != values.shape:
                raise ValueError("age channel is required and must match values")
            channels.append(age)
        fused = torch.cat(channels, dim=-1)
        patches = fused.reshape(batch, time // self.patch_size, self.patch_size * fused.shape[-1])
        return self.projection(patches)

