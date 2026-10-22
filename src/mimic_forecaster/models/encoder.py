"""Transformer time-series encoder."""

from __future__ import annotations

import torch
from torch import nn


class TimeSeriesEncoder(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, ffn_dim: int, dropout: float,
                 max_patches: int) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, max_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.position, std=0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, count, _ = patches.shape
        if count + 1 > self.position.shape[1]:
            raise ValueError("too many patches for positional embedding")
        sequence = torch.cat([self.cls.expand(batch, -1, -1), patches], dim=1)
        encoded = self.transformer(sequence + self.position[:, :count + 1])
        return self.norm(encoded[:, 0])

