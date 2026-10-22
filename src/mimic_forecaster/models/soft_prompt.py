"""Soft prompt parameterization and optional CMA-ES optimization."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SoftPrompt(nn.Module):
    def __init__(self, n_tokens: int, hidden_dim: int, init_std: float = 0.01) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(n_tokens, hidden_dim), requires_grad=False)
        nn.init.normal_(self.embedding, std=init_std)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.embedding.unsqueeze(0).expand(batch_size, -1, -1)


class RandomSubspacePrompt:
    def __init__(self, prompt_shape: tuple[int, int], subspace_dim: int, seed: int) -> None:
        size = int(np.prod(prompt_shape)); rng = np.random.default_rng(seed)
        self.shape = prompt_shape
        self.projection = rng.normal(0, 1 / np.sqrt(subspace_dim), (size, subspace_dim)).astype("float32")

    def decode(self, coordinates: np.ndarray, base: np.ndarray) -> np.ndarray:
        return base + (self.projection @ np.asarray(coordinates, dtype="float32")).reshape(self.shape)

